#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def require(path, failures, kind="path"):
    if not path.exists():
        failures.append(f"missing {kind}: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get("DETECTRON2_DATASETS", "datasets"),
        help="Dataset root; defaults to DETECTRON2_DATASETS or ./datasets",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "voc", "coco"),
        default="all",
        help="Validate all data or only the dataset needed for a quick run.",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    failures = []
    print(f"dataset_root={root}")

    if args.dataset in {"all", "voc"}:
        for year in ("VOC2007", "VOC2012"):
            base = root / year
            for relative in ("Annotations", "ImageSets/Main", "JPEGImages"):
                require(base / relative, failures)
        for seed in (0, 1, 2):
            voc = root / f"vocsplit/seed{seed}"
            require(voc, failures)
            if voc.is_dir():
                count = len(list(voc.glob("box_*shot_*_train.txt")))
                print(f"voc seed{seed}: {count} support files")
                if count != 100:
                    failures.append(
                        f"voc seed{seed}: expected 100 support files, got {count}"
                    )

    if args.dataset in {"all", "coco"}:
        require(root / "coco/trainval2014", failures)
        require(root / "coco/val2014", failures)
        for seed in (0, 1, 2):
            coco = root / f"cocosplit/seed{seed}"
            require(coco, failures)
            if coco.is_dir():
                count = len(list(coco.glob("full_box_*shot_*_trainval.json")))
                print(f"coco seed{seed}: {count} support files")
                if count != 480:
                    failures.append(
                        f"coco seed{seed}: expected 480 support files, got {count}"
                    )
        datasplit = root / "cocosplit/datasplit"
        for name in ("trainvalno5k.json", "5k.json"):
            path = datasplit / name
            require(path, failures, "COCO split JSON")

    if failures:
        print("\nData check: FAIL")
        for item in failures:
            print(" - " + item)
        return 2
    print("\nData check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

