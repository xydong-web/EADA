#!/usr/bin/env python3
"""Measure novel-GT coverage by top-K RPN proposals for SAIA diagnostics."""

import argparse
import json
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import Boxes, pairwise_iou

from eada.config import add_eada_config
import eada.data.builtin  # noqa: F401
import eada.model  # noqa: F401
from train_net import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--saia", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    cfg = get_cfg()
    add_eada_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.MODEL.EADA.SAIA_ENABLED = bool(args.saia)
    cfg.MODEL.EADA.SAIA_ALPHA = float(args.alpha)
    cfg.TEST.SPDI_ENABLED = False
    cfg.DATASETS.TEST = ("voc_2007_test_all1",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(args.checkpoint, resume=False)
    model.eval()

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    novel_ids = {
        metadata.thing_classes.index(name) for name in metadata.novel_classes
    }
    ground_truth = {}
    for record in DatasetCatalog.get(cfg.DATASETS.TEST[0]):
        boxes = []
        for annotation in record.get("annotations", []):
            if int(annotation["category_id"]) in novel_ids:
                boxes.append(annotation["bbox"])
        ground_truth[str(record["image_id"])] = boxes

    captured = {}

    def hook(module, inputs, output):
        del module, inputs
        captured["proposals"] = output[0] if isinstance(output, tuple) else output

    handle = model.proposal_generator.register_forward_hook(hook)
    loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    total = hit50 = hit75 = 0
    best_values = []
    images = 0
    with torch.no_grad():
        for batch in loader:
            captured.clear()
            model(batch)
            proposals = captured.get("proposals")
            if proposals is None:
                raise RuntimeError("RPN hook did not capture proposals")
            for item, proposal in zip(batch, proposals):
                raw_boxes = ground_truth.get(str(item["image_id"]), [])
                if not raw_boxes:
                    continue
                gt = torch.tensor(raw_boxes, dtype=torch.float32, device=proposal.proposal_boxes.tensor.device)
                original_h = float(item.get("height", proposal.image_size[0]))
                original_w = float(item.get("width", proposal.image_size[1]))
                resized_h, resized_w = proposal.image_size
                scale = torch.tensor(
                    [original_w / resized_w, original_h / resized_h,
                     original_w / resized_w, original_h / resized_h],
                    device=gt.device,
                )
                order = proposal.objectness_logits.argsort(descending=True)[: args.top_k]
                boxes = proposal.proposal_boxes.tensor[order] * scale
                if len(boxes):
                    best = pairwise_iou(Boxes(gt), Boxes(boxes)).max(dim=1).values
                else:
                    best = torch.zeros(len(gt), device=gt.device)
                total += len(gt)
                hit50 += int((best >= 0.50).sum())
                hit75 += int((best >= 0.75).sum())
                best_values.extend(float(value) for value in best.cpu())
            images += len(batch)
    handle.remove()

    result = {
        "dataset": cfg.DATASETS.TEST[0],
        "checkpoint": args.checkpoint,
        "saia": bool(args.saia),
        "alpha": float(args.alpha),
        "top_k": int(args.top_k),
        "images": images,
        "novel_gt": total,
        "recall_at_100_iou50_percent": 100.0 * hit50 / max(total, 1),
        "recall_at_100_iou75_percent": 100.0 * hit75 / max(total, 1),
        "mean_best_iou_percent": 100.0 * sum(best_values) / max(len(best_values), 1),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

