#!/usr/bin/env python3
"""Profile EADA detector/SPDI inference cost with batch size one."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.utils.analysis import flop_count_operators
from fvcore.nn import FlopCountAnalysis

from eada.config import add_eada_config
import eada.data.builtin  # noqa: F401
import eada.model  # noqa: F401
from train_net import Trainer


MEMORY_RE = re.compile(r"max_mem:\s*([0-9]+)M")


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode_image(images)


def peak_training_memory(path):
    if not path:
        return None
    values = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = MEMORY_RE.search(line)
        if match:
            values.append(int(match.group(1)))
    return max(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="voc_2007_test_all1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--spdi", action="store_true")
    parser.add_argument("--disable-saia", action="store_true")
    parser.add_argument("--clip-checkpoint", default="")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--training-log", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the efficiency profile")

    cfg = get_cfg()
    add_eada_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.checkpoint
    if args.disable_saia:
        cfg.MODEL.EADA.SAIA_ENABLED = False
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.SPDI_ENABLED = bool(args.spdi)
    cfg.TEST.SPDI_TOPK = int(args.top_k)
    if args.clip_checkpoint:
        cfg.TEST.SPDI_CHECKPOINT = args.clip_checkpoint
    cfg.freeze()

    model = Trainer.build_model(cfg).eval()
    DetectionCheckpointer(model).resume_or_load(args.checkpoint, resume=False)
    loader = build_detection_test_loader(cfg, args.dataset)
    first = next(iter(loader))

    # Detector FLOPs are measured without SPDI because the frozen CLIP image
    # encoder is profiled separately and multiplied by the proposal budget.
    spdi_state = model.roi_heads.spdi_enabled
    model.roi_heads.spdi_enabled = False
    with torch.no_grad():
        detector_gflops = float(sum(flop_count_operators(model, first).values()))
    model.roi_heads.spdi_enabled = spdi_state

    detector_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    clip_parameters = 0
    clip_gflops_per_crop = 0.0
    if args.spdi:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=args.clip_checkpoint or "openai"
        )
        clip_model = clip_model.cuda().eval()
        clip_parameters = int(sum(parameter.numel() for parameter in clip_model.parameters()))
        dummy = torch.zeros(1, 3, 224, 224, device="cuda")
        with torch.no_grad():
            clip_gflops_per_crop = float(
                FlopCountAnalysis(ImageEncoder(clip_model), dummy)
                .unsupported_ops_warnings(False)
                .uncalled_modules_warnings(False)
                .total()
                / 1e9
            )
        del clip_model, dummy
        torch.cuda.empty_cache()

    samples = []
    for batch in build_detection_test_loader(cfg, args.dataset):
        samples.append(batch)
        if len(samples) >= args.warmup + args.images:
            break
    if len(samples) <= args.warmup:
        raise RuntimeError("Not enough evaluation samples for profiling")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in samples[: args.warmup]:
            model(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    measured = samples[args.warmup : args.warmup + args.images]
    with torch.no_grad():
        for batch in measured:
            model(batch)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    count = sum(len(batch) for batch in measured)
    latency_ms = 1000.0 * seconds / count

    payload = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "spdi": bool(args.spdi),
        "saia": not bool(args.disable_saia),
        "top_k": args.top_k if args.spdi else 0,
        "detector_parameters": detector_parameters,
        "clip_parameters": clip_parameters,
        "detector_gflops_per_image": detector_gflops,
        "clip_image_encoder_gflops_per_crop": clip_gflops_per_crop,
        "upper_bound_gflops_per_image": detector_gflops
        + (args.top_k * clip_gflops_per_crop if args.spdi else 0.0),
        "inference_peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "latency_ms_per_image": latency_ms,
        "fps": 1000.0 / latency_ms,
        "measured_images": count,
        "training_peak_memory_mib_from_log": peak_training_memory(args.training_log),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

