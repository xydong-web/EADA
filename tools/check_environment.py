#!/usr/bin/env python3
import importlib
import importlib.metadata
import sys


EXPECTED = {
    "python": "3.9",
    "torch": "2.2.2",
    "torchvision": "0.17.2",
    "detectron2": "0.3",
    "mmcv": "1.7.2",
    "mmdet": "2.28.2",
    "numpy": "1.26.4",
}


def version_of(name):
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


def main():
    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": version_of("torch"),
        "torchvision": version_of("torchvision"),
        "detectron2": version_of("detectron2"),
        "mmcv": version_of("mmcv"),
        "mmdet": version_of("mmdet"),
        "numpy": version_of("numpy"),
        "open_clip": importlib.metadata.version("open-clip-torch"),
    }
    import torch
    import open_clip
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
    del MultiScaleDeformableAttention

    failures = []
    expected_versions = dict(EXPECTED)
    expected_versions["open_clip"] = "2.24.0"
    for key, expected in expected_versions.items():
        value = actual[key]
        ok = value.startswith(expected)
        print(f"{key:12s} {value:20s} expected {expected:8s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((key, value, expected))
    print(f"cuda_build   {torch.version.cuda}")
    print(f"cuda_runtime {'available' if torch.cuda.is_available() else 'not available'}")
    if failures:
        print("\nEnvironment differs from the validated paper stack.")
        return 2
    print("\nEnvironment check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

