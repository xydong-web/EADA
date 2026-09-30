#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
from pathlib import Path


VOC_PATTERN = re.compile(r"split(?P<split>[123])/(?P<shot>1|2|3|5|10)shot/seed(?P<seed>[012])")
COCO_PATTERN = re.compile(r"(?P<shot>1|2|3|5|10|30)shot/seed(?P<seed>[012])")


def bbox_metrics(path):
    payload = json.loads(path.read_text())
    if "bbox" in payload:
        return payload["bbox"]
    if "metrics" in payload and "bbox" in payload["metrics"]:
        return payload["metrics"]["bbox"]
    raise ValueError(f"No bbox metrics in {path}")


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pattern = VOC_PATTERN if args.dataset == "voc" else COCO_PATTERN
    groups = {}
    for path in args.root.rglob("evaluation_results.json"):
        relative = path.relative_to(args.root).as_posix()
        if "/eval/" not in "/" + relative:
            continue
        match = pattern.search(relative)
        if not match:
            continue
        info = {key: int(value) for key, value in match.groupdict().items()}
        key = (
            f"split{info['split']}_{info['shot']}shot"
            if args.dataset == "voc"
            else f"{info['shot']}shot"
        )
        metrics = bbox_metrics(path)
        if "nAP50" not in metrics:
            raise ValueError(f"Missing nAP50 in {path}")
        groups.setdefault(key, []).append(
            {"seed": info["seed"], "path": str(path), **metrics}
        )

    cells = {}
    for key, runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda item: item["seed"])
        best = max(runs, key=lambda item: (item["nAP50"], item.get("nAP", -math.inf)))
        metric_names = sorted(
            name for name, value in best.items()
            if name not in {"seed", "path"} and isinstance(value, (int, float))
        )
        cells[key] = {
            "selected_seed": best["seed"],
            "selected": {name: best[name] for name in metric_names},
            "runs": runs,
            "mean_std": {
                name: summarize([float(run[name]) for run in runs if name in run])
                for name in metric_names
            },
        }

    payload = {
        "dataset": args.dataset,
        "selection_metric": "nAP50",
        "selection_tiebreaker": "nAP",
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

