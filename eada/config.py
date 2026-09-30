from detectron2.config import CfgNode as CN


def add_eada_config(cfg):
    cfg.MODEL.EADA = CN()
    cfg.MODEL.EADA.SAIA_ENABLED = True
    cfg.MODEL.EADA.SAIA_ALPHA = 0.2
    cfg.MODEL.EADA.DC_ENABLED = True
    # Detector-head protocol used by the experiment matrix.
    # ``residual_trainable`` reproduces base pretraining: a learned residual
    # mapping precedes a trainable classifier. ``residual_frozen`` is the
    # VOC few-shot protocol, which keeps the mapping active but freezes the
    # expanded classifier. ``standard`` is the COCO few-shot protocol.
    cfg.MODEL.EADA.ROI_PREDICTOR_MODE = "standard"
    cfg.MODEL.EADA.RPN_GRADIENT_SCALE = 0.0
    cfg.MODEL.EADA.ROI_GRADIENT_SCALE = 0.001
    cfg.MODEL.EADA.CLS_DROPOUT = 0.8
    cfg.MODEL.EADA.FREEZE_ROI_FEATURE = True
    cfg.MODEL.EADA.FSIS_MASK_ONLY = False
    cfg.MODEL.EADA.FSIS_FEATURE_SOURCE = "pre_saia"
    cfg.DATASETS.EADA_TWO_STREAM = False
    cfg.DATASETS.BASE_TRAIN = ()
    cfg.TEST.SPDI_ENABLED = True
    cfg.TEST.SPDI_CHECKPOINT = ""
    cfg.TEST.SPDI_MODEL_NAME = "ViT-B-16"
    cfg.TEST.SPDI_PRETRAINED = "openai"
    cfg.TEST.SPDI_TOPK = 100
    cfg.TEST.SPDI_TEMPERATURE = 0.01
    cfg.TEST.SPDI_LAMBDA_NOVEL = 0.7
    cfg.TEST.SPDI_FUSION_MODE = "absolute"
    cfg.TEST.SPDI_PLACEMENT = "pre_nms"
    cfg.TEST.SPDI_BATCH_SIZE = 64
    cfg.TEST.SPDI_TEXT_TEMPLATES = [
        "a photo of a {}",
        "a clean photo of a {}",
    ]
