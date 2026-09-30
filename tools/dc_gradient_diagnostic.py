#!/usr/bin/env python3
"""Compare CE and DC gradients on the same sampled background RoIs."""

import argparse
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.events import EventStorage

from eada.config import add_eada_config
from eada.losses import dc_logits
import eada.data.builtin  # noqa: F401
import eada.model  # noqa: F401
from train_net import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=64)
    args = parser.parse_args()

    cfg = get_cfg()
    add_eada_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.MODEL.EADA.SAIA_ENABLED = False
    cfg.MODEL.EADA.DC_ENABLED = False
    cfg.DATASETS.TRAIN = ("removevoc_2007_trainval_all1_1shot_seed0",)
    cfg.DATASETS.EADA_TWO_STREAM = False
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.SPDI_ENABLED = False
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(args.checkpoint, resume=False)
    model.train()
    loader = Trainer.build_train_loader(cfg)

    captured = {"logits": None, "proposals": None}

    def cls_hook(module, inputs, output):
        del module, inputs
        captured["logits"] = output

    handle = model.roi_heads.box_predictor.cls_score.register_forward_hook(cls_hook)
    original = model.roi_heads.label_and_sample_proposals

    def wrapped(instance, proposals, targets):
        sampled = original(proposals, targets)
        captured["proposals"] = sampled
        return sampled

    model.roi_heads.label_and_sample_proposals = types.MethodType(wrapped, model.roi_heads)

    totals = dict(ce_absent=0.0, dc_absent=0.0, ce_supported=0.0, dc_supported=0.0)
    background_rois = images = 0
    for batch in loader:
        if images >= args.max_images:
            break
        captured["logits"] = None
        captured["proposals"] = None
        with torch.no_grad(), EventStorage():
            model(batch)
        logits = captured["logits"]
        proposals = captured["proposals"]
        if logits is None or proposals is None:
            raise RuntimeError("Failed to capture logits/proposals")
        offset = 0
        for item, proposal in zip(batch, proposals):
            count = len(proposal)
            x = logits[offset:offset + count].detach().float().requires_grad_(True)
            y = proposal.gt_classes.to(x.device)
            background = x.shape[1] - 1
            rows = (y == background).nonzero(as_tuple=False).squeeze(1)
            if rows.numel():
                observed = sorted(
                    set(int(v) for v in item["instances"].gt_classes.unique().tolist())
                )
                absent = [c for c in range(background) if c not in observed]
                supported = [c for c in range(background) if c in observed]
                ce = F.cross_entropy(x, y, reduction="sum")
                ce_grad = torch.autograd.grad(ce, x, retain_graph=True)[0]
                gated = dc_logits(x, y, [count], [observed])
                dc = F.cross_entropy(gated, y, reduction="sum")
                dc_grad = torch.autograd.grad(dc, x)[0]
                if absent:
                    totals["ce_absent"] += float(ce_grad[rows][:, absent].abs().sum())
                    totals["dc_absent"] += float(dc_grad[rows][:, absent].abs().sum())
                if supported:
                    totals["ce_supported"] += float(ce_grad[rows][:, supported].abs().sum())
                    totals["dc_supported"] += float(dc_grad[rows][:, supported].abs().sum())
                background_rois += int(rows.numel())
            offset += count
        images += len(batch)

    handle.remove()
    model.roi_heads.label_and_sample_proposals = original
    scale = 1000.0 / max(background_rois, 1)
    result = {
        "images": images,
        "background_rois": background_rois,
        "absent_class_gradient_x1e3": {
            "control_ce": totals["ce_absent"] * scale,
            "dc": totals["dc_absent"] * scale,
        },
        "supported_class_gradient_x1e3": {
            "control_ce": totals["ce_supported"] * scale,
            "dc": totals["dc_supported"] * scale,
        },
        "checkpoint": args.checkpoint,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

