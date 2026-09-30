import torch
from torch import nn
import torch.nn.functional as functional
import fvcore.nn.weight_init as weight_init
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, ROI_HEADS_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.data import MetadataCatalog
from .losses import dc_loss
from .saia import initialize_saia, apply_saia
from .spdi import SPDI


def decouple(features, scale):
    return features.detach() + scale * (features - features.detach())


class AffineLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features):
        return features * self.weight + self.bias


@ROI_HEADS_REGISTRY.register()
class EADAROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.fc_s = nn.Linear(2048, 2048)
        self.fc_l = nn.Linear(2048, 2048)
        for layer in (self.fc_s, self.fc_l):
            weight_init.c2_xavier_fill(layer)
        self.roi_predictor_mode = str(cfg.MODEL.EADA.ROI_PREDICTOR_MODE)
        if self.roi_predictor_mode in {"residual_trainable", "residual_frozen"}:
            # Base/VOC checkpoints contain this learned residual mapping before
            # the linear classifier.
            self.box_predictor.mapping = nn.Linear(2048, 2048, bias=False)
            self.box_predictor.cls_score.requires_grad_(
                self.roi_predictor_mode == "residual_trainable"
            )
        elif self.roi_predictor_mode == "standard":
            # COCO and base-training checkpoints use the ordinary trainable
            # Fast R-CNN linear classifier and do not contain a mapping tensor.
            self.box_predictor.cls_score.requires_grad_(True)
        else:
            raise ValueError(
                "MODEL.EADA.ROI_PREDICTOR_MODE must be standard, "
                "residual_trainable, or residual_frozen"
            )
        self.dc_enabled = cfg.MODEL.EADA.DC_ENABLED
        self.dropout = cfg.MODEL.EADA.CLS_DROPOUT
        self.mask_only_training = bool(cfg.MODEL.EADA.FSIS_MASK_ONLY)
        self.fsis_feature_source = str(cfg.MODEL.EADA.FSIS_FEATURE_SOURCE)
        self.spdi_enabled = cfg.TEST.SPDI_ENABLED
        self.spdi_checkpoint = cfg.TEST.SPDI_CHECKPOINT
        self.spdi_model_name = cfg.TEST.SPDI_MODEL_NAME
        self.spdi_pretrained = cfg.TEST.SPDI_PRETRAINED
        self.spdi_topk = cfg.TEST.SPDI_TOPK
        self.spdi_temperature = cfg.TEST.SPDI_TEMPERATURE
        self.spdi_lambda_novel = cfg.TEST.SPDI_LAMBDA_NOVEL
        self.spdi_fusion_mode = str(cfg.TEST.SPDI_FUSION_MODE)
        self.spdi_placement = str(cfg.TEST.SPDI_PLACEMENT)
        if self.spdi_placement not in {"pre_nms", "post_nms"}:
            raise ValueError("TEST.SPDI_PLACEMENT must be pre_nms or post_nms")
        self.spdi_batch_size = cfg.TEST.SPDI_BATCH_SIZE
        self.spdi_text_templates = tuple(cfg.TEST.SPDI_TEXT_TEMPLATES)
        self.spdi = None
        self.input_format = cfg.INPUT.FORMAT
        dataset = (cfg.DATASETS.TEST or cfg.DATASETS.TRAIN)[0]
        metadata = MetadataCatalog.get(dataset)
        self.class_names = list(metadata.thing_classes)
        self.novel_ids = [index for index, name in enumerate(self.class_names) if name in metadata.novel_classes]
        if len(self.class_names) != self.num_classes:
            raise ValueError("Dataset metadata and classifier disagree")
        if self.mask_on:
            feature_name = self.in_features[0]
            resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            self.mask_pooler = ROIPooler(
                output_size=resolution,
                scales=(1.0 / input_shape[feature_name].stride,),
                sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
                pooler_type=cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE,
            )
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(
                    channels=input_shape[feature_name].channels,
                    width=resolution,
                    height=resolution,
                ),
            )

    def _forward_mask(self, features, instances, training):
        boxes = [
            item.proposal_boxes if training else item.pred_boxes
            for item in instances
        ]
        pooled = self.mask_pooler(
            [features[name] for name in self.in_features],
            boxes,
        )
        return self.mask_head(pooled, instances)

    def forward(self, images, features, proposals, targets=None, mask_features=None):
        observed = None
        if self.training:
            observed = [target.gt_classes.unique().tolist() for target in targets]
            proposals = self.label_and_sample_proposals(proposals, targets)
            if self.mask_on and self.mask_only_training:
                foreground, _ = select_foreground_proposals(
                    proposals,
                    self.num_classes,
                )
                source = mask_features if mask_features is not None else features
                return foreground, self._forward_mask(source, foreground, True)
        pooled = self._shared_roi_transform([features[name] for name in self.in_features], [proposal.proposal_boxes for proposal in proposals]).mean(dim=(2, 3))
        semantic = functional.relu(self.fc_s(pooled))
        localization = functional.relu(self.fc_l(pooled))
        semantic = functional.dropout(
            semantic,
            self.dropout,
            training=self.training,
        )
        if self.roi_predictor_mode in {"residual_trainable", "residual_frozen"}:
            semantic = self.box_predictor.mapping(semantic) + semantic
        predictions = (
            self.box_predictor.cls_score(semantic),
            self.box_predictor.bbox_pred(localization),
        )
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            if self.dc_enabled:
                labels = torch.cat([proposal.gt_classes for proposal in proposals])
                losses["loss_cls"] = dc_loss(predictions[0], labels, [len(proposal) for proposal in proposals], observed)
            if self.mask_on:
                foreground, _ = select_foreground_proposals(
                    proposals,
                    self.num_classes,
                )
                source = mask_features if mask_features is not None else features
                losses.update(self._forward_mask(source, foreground, True))
            return proposals, losses
        boxes = self.box_predictor.predict_boxes(predictions, proposals)
        scores = self.box_predictor.predict_probs(predictions, proposals)
        if self.spdi_enabled:
            if self.spdi is None:
                self.spdi = SPDI(
                    self.class_names,
                    self.novel_ids,
                    predictions[0].device,
                    checkpoint=self.spdi_checkpoint,
                    model_name=self.spdi_model_name,
                    pretrained=self.spdi_pretrained,
                    prompts=self.spdi_text_templates,
                    topk=self.spdi_topk,
                    temperature=self.spdi_temperature,
                    detector_weight=self.spdi_lambda_novel,
                    fusion_mode=self.spdi_fusion_mode,
                    batch_size=self.spdi_batch_size,
                )
            if self.spdi_placement == "pre_nms":
                calibrated = []
                for index, (image_height, image_width) in enumerate(images.image_sizes):
                    image = images.tensor[index, :, :image_height, :image_width]
                    if self.input_format == "BGR":
                        image = image.flip(0)
                    calibrated.append(self.spdi(image, boxes[index], scores[index]))
                scores = calibrated
        instances, _ = fast_rcnn_inference(boxes, scores, [proposal.image_size for proposal in proposals], self.box_predictor.test_score_thresh, self.box_predictor.test_nms_thresh, self.box_predictor.test_topk_per_image)
        if self.spdi_enabled and self.spdi_placement == "post_nms":
            rescored = []
            for index, (image_height, image_width) in enumerate(images.image_sizes):
                image = images.tensor[index, :, :image_height, :image_width]
                if self.input_format == "BGR":
                    image = image.flip(0)
                rescored.append(self.spdi.rescore_instances(image, instances[index]))
            instances = rescored
        if self.mask_on:
            source = mask_features if mask_features is not None else features
            instances = self._forward_mask(source, instances, False)
        return instances, {}


@META_ARCH_REGISTRY.register()
class EADA(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.MODEL.KEYPOINT_ON or cfg.MODEL.RESNETS.DEPTH != 101:
            raise ValueError("EADA release supports R101-C4 detection/instance segmentation")
        shape = self.backbone.output_shape()
        if list(shape) != ["res4"] or shape["res4"].channels != 1024:
            raise ValueError("EADA requires standard R101-C4 res4")
        self.saia_enabled = cfg.MODEL.EADA.SAIA_ENABLED
        self.saia_alpha = cfg.MODEL.EADA.SAIA_ALPHA
        self.rpn_scale = cfg.MODEL.EADA.RPN_GRADIENT_SCALE
        self.roi_scale = cfg.MODEL.EADA.ROI_GRADIENT_SCALE
        self.fsis_mask_only = bool(cfg.MODEL.EADA.FSIS_MASK_ONLY)
        self.fsis_feature_source = str(cfg.MODEL.EADA.FSIS_FEATURE_SOURCE)
        if self.fsis_feature_source not in {"pre_saia", "post_saia"}:
            raise ValueError("FSIS_FEATURE_SOURCE must be pre_saia or post_saia")
        self.affine_rpn = AffineLayer(1024)
        self.affine_rcnn = AffineLayer(1024)
        if self.saia_enabled:
            initialize_saia(self, shape["res4"].channels)
        if bool(cfg.MODEL.EADA.FREEZE_ROI_FEATURE):
            for parameter in self.roi_heads.res5.parameters():
                parameter.requires_grad_(False)
        if self.fsis_mask_only:
            if not cfg.MODEL.MASK_ON:
                raise ValueError("FSIS_MASK_ONLY requires MODEL.MASK_ON=True")
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.roi_heads.mask_head.parameters():
                parameter.requires_grad_(True)
        self.to(cfg.MODEL.DEVICE)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        targets = [item["instances"].to(self.device) for item in batched_inputs]
        raw_features, features = self._features(images)
        mask_features = {
            "res4": raw_features if self.fsis_feature_source == "pre_saia" else features
        }
        proposals, proposal_losses = self.proposal_generator(images, {"res4": self.affine_rpn(decouple(features, self.rpn_scale))}, targets)
        _, losses = self.roi_heads(
            images,
            {"res4": self.affine_rcnn(decouple(features, self.roi_scale))},
            proposals,
            targets,
            mask_features,
        )
        if not self.fsis_mask_only:
            losses.update(proposal_losses)
        return losses

    def _features(self, images):
        raw_features = self.backbone(images.tensor)["res4"]
        features = (
            apply_saia(self, raw_features, images)
            if self.saia_enabled
            else raw_features
        )
        return raw_features, features

    @torch.no_grad()
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        from detectron2.structures import ImageList
        if detected_instances is not None:
            raise ValueError("EADA uses RPN proposals")
        images = self.preprocess_image(batched_inputs)
        raw_features, features = self._features(images)
        mask_features = {
            "res4": raw_features if self.fsis_feature_source == "pre_saia" else features
        }
        proposals, _ = self.proposal_generator(images, {"res4": self.affine_rpn(features)}, None)
        raw_images = ImageList.from_tensors([item["image"].to(self.device).float() for item in batched_inputs], self.backbone.size_divisibility)
        results, _ = self.roi_heads(
            raw_images,
            {"res4": self.affine_rcnn(features)},
            proposals,
            mask_features=mask_features,
        )
        return self._postprocess(results, batched_inputs, images.image_sizes) if do_postprocess else results
