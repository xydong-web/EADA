#!/usr/bin/env python3
import argparse
from pathlib import Path


PAPER_INITIALIZATIONS = {
    "voc_split1_all.pth": 259869626,
    "voc_split2_all.pth": 259869626,
    "voc_split3_all.pth": 259869626,
    "coco_all.pth": 262328442,
    "coco_base_mask_init.pth": 280699296,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("weights"))
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Require only the four FSOD detector initializations.",
    )
    args = parser.parse_args()
    expected = dict(PAPER_INITIALIZATIONS)
    if args.detection_only:
        expected.pop("coco_base_mask_init.pth")

    failures = []
    for name, expected_size in expected.items():
        path = args.root / name
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        size = path.stat().st_size
        ok = size == expected_size
        print(
            f"{name:30s} size={size:10d} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures.append(name)
    if failures:
        print("\nWeight check: FAIL")
        for item in failures:
            print(" - " + item)
        return 2
    print("\nWeight check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

