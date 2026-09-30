#!/usr/bin/env python3
"""Compose an EADA detector checkpoint with a class-agnostic mask head."""

import argparse
from pathlib import Path
import torch


MASK_PREFIX = "roi_heads.mask_head."


def merge_states(detector_state, mask_state):
    result = {
        key: value
        for key, value in detector_state.items()
        if not key.startswith(MASK_PREFIX)
    }
    mask_keys = sorted(key for key in mask_state if key.startswith(MASK_PREFIX))
    if not mask_keys:
        raise ValueError("Mask checkpoint does not contain an EADA mask head")
    for key in mask_keys:
        result[key] = mask_state[key]
    return result, mask_keys


def main():
    parser = argparse.ArgumentParser(
        description="Attach a base-trained class-agnostic mask head to EADA"
    )
    parser.add_argument("detector")
    parser.add_argument("mask_checkpoint")
    parser.add_argument("destination")
    args = parser.parse_args()

    detector = torch.load(args.detector, map_location="cpu")
    mask_checkpoint = torch.load(args.mask_checkpoint, map_location="cpu")
    merged, mask_keys = merge_states(
        detector["model"],
        mask_checkpoint["model"],
    )
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": merged,
            "iteration": -1,
            "eada_fsis": {
                "mask_parameter_count": len(mask_keys),
                "detector_checkpoint": str(args.detector),
                "mask_checkpoint": str(args.mask_checkpoint),
            },
        },
        destination,
    )
    print(destination)


if __name__ == "__main__":
    main()
