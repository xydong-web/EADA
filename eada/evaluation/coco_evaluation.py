import os
import io
import json
import copy
import torch
import logging
import itertools
import contextlib
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from detectron2.structures import BoxMode
from detectron2.utils import comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import create_small_table
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import DatasetEvaluator


class COCOEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, distributed, output_dir=None):

        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._has_masks = False

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")
            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
            or "novel" in dataset_name
        self._base_classes = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self._novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                               44, 62, 63, 64, 67, 72]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []
        self._has_masks = False

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                has_masks = instances.has("pred_masks")
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
                # Keep this bit separately so an image with zero detections
                # can still select the segm evaluator when the model emitted
                # an empty pred_masks field.
                prediction["has_masks"] = has_masks
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))
            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._has_masks = any(
                prediction.get("has_masks", False)
                for prediction in self._predictions
            )
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        if self._has_masks:
            missing_masks = [
                index for index, result in enumerate(self._coco_results)
                if "segmentation" not in result
            ]
            if missing_masks:
                raise ValueError(
                    "pred_masks were present for only part of the predictions; "
                    "cannot run a consistent COCO segm evaluation (missing "
                    "indices: {})".format(missing_masks[:5])
                )

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()
            if self._has_masks:
                segm_path = os.path.join(
                    self._output_dir, "coco_instances_results_segm.json"
                )
                with PathManager.open(segm_path, "w") as f:
                    f.write(json.dumps([
                        result for result in self._coco_results
                        if "segmentation" in result
                    ]))
                    f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        task_types = ["bbox"]
        if self._has_masks:
            task_types.append("segm")

        if self._is_splits:
            split_specs = [
                ("all", None, self._metadata.get("thing_classes")),
                ("base", self._base_classes, self._metadata.get("base_classes")),
                ("novel", self._novel_classes, self._metadata.get("novel_classes")),
            ]
            for iou_type in task_types:
                self._results[iou_type] = {}
                for split, classes, names in split_specs:
                    if "all" not in self._dataset_name and \
                            split not in self._dataset_name:
                        continue
                    coco_eval = (
                        _evaluate_predictions_on_coco(
                            self._coco_api,
                            self._coco_results,
                            iou_type,
                            classes,
                        )
                        if len(self._coco_results) > 0
                        else None  # cocoapi does not handle empty results very well
                    )
                    res_ = self._derive_coco_results(
                        coco_eval, iou_type, class_names=names
                    )
                    res = {}
                    for metric in res_.keys():
                        if len(metric) <= 4:
                            if split == "all":
                                res[metric] = res_[metric]
                            elif split == "base":
                                res["b" + metric] = res_[metric]
                            elif split == "novel":
                                res["n" + metric] = res_[metric]
                    self._results[iou_type].update(res)

                # Add "AP" if the dataset selects only base or novel classes.
                if "AP" not in self._results[iou_type]:
                    if "nAP" in self._results[iou_type]:
                        self._results[iou_type]["AP"] = self._results[iou_type]["nAP"]
                    else:
                        self._results[iou_type]["AP"] = self._results[iou_type]["bAP"]
        else:
            for iou_type in task_types:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, iou_type,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                self._results[iou_type] = self._derive_coco_results(
                    coco_eval,
                    iou_type,
                    class_names=self._metadata.get("thing_classes"),
                )

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
                for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
                create_small_table(results)
        )

        if class_names is None:
            return results
        # Compute per-category AP
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        params = getattr(coco_eval, "params", None)
        iou_thresholds = np.asarray(
            getattr(params, "iouThrs", np.linspace(0.50, 0.95, precisions.shape[0]))
        )

        def mean_valid(values):
            values = values[values > -1]
            return float(np.mean(values) * 100) if values.size else float("nan")

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            ap = mean_valid(precision)
            ap50 = mean_valid(precision[np.isclose(iou_thresholds, 0.50)])
            ap75 = mean_valid(precision[np.isclose(iou_thresholds, 0.75)])
            results_per_category.append((str(name), ap, ap50, ap75))

        # tabulate it
        display_rows = [(name, ap) for name, ap, _, _ in results_per_category]
        N_COLS = min(6, len(display_rows) * 2)
        results_flatten = list(itertools.chain(*display_rows))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        for name, ap, ap50, ap75 in results_per_category:
            results["AP-" + name] = ap
            results["AP50-" + name] = ap50
            results["AP75-" + name] = ap75
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks
        if torch.is_tensor(masks):
            masks = masks.numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4:
            if masks.shape[1] != 1:
                raise ValueError(
                    "COCO serialization expects class-agnostic pred_masks "
                    "with shape (N, 1, H, W), got {}".format(masks.shape)
                )
            masks = masks[:, 0]
        if masks.ndim != 3 or masks.shape[0] != num_instance:
            raise ValueError(
                "pred_masks must have shape (N, H, W) or (N, 1, H, W), "
                "got {} for {} instances".format(masks.shape, num_instance)
            )

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if masks is not None:
            # COCO RLE expects a Fortran-contiguous uint8 bitmap. The model
            # emits probabilities, so threshold before encoding rather than
            # treating every non-zero floating-point value as foreground.
            bitmap = np.asarray(masks[k] >= 0.5, order="F", dtype=np.uint8)
            rle = mask_util.encode(bitmap[:, :, None])[0]
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("ascii")
            result["segmentation"] = rle
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    # Some official cross-domain COCO annotations omit the optional top-level
    # ``info`` field, while older pycocotools releases access it unconditionally
    # in COCO.loadRes(). Keep the source JSON untouched and normalize only the
    # in-memory API object used for evaluation.
    coco_gt.dataset.setdefault("info", {})
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
