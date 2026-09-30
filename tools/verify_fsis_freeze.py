#!/usr/bin/env python3
"""Verify that EADA FSIS training changed only mask-head parameters."""

import argparse
import json
from pathlib import Path
import torch


MASK_PREFIX = "roi_heads.mask_head."


def compare(reference_path, candidate_path):
    reference = torch.load(reference_path, map_location="cpu")["model"]
    candidate = torch.load(candidate_path, map_location="cpu")["model"]
    keys = sorted(set(reference) | set(candidate))
    changed = []
    missing = []
    unexpected = []
    for key in keys:
        if key.startswith(MASK_PREFIX):
            continue
        if key not in candidate:
            missing.append(key)
        elif key not in reference:
            unexpected.append(key)
        elif not torch.equal(reference[key], candidate[key]):
            changed.append(key)
    return {
        "status": "PASS" if not changed and not missing and not unexpected else "FAIL",
        "changed_non_mask_parameters": changed,
        "missing_non_mask_parameters": missing,
        "unexpected_non_mask_parameters": unexpected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
