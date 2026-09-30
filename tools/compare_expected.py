#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--table",
        choices=("table1_pascal_voc", "table2_ms_coco", "table4_generalized_voc"),
        required=True,
    )
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Compare only completed cells; useful for the reviewer quick check.",
    )
    args = parser.parse_args()
    actual = json.loads(args.results.read_text())
    expected = json.loads((ROOT / "expected/paper_results.json").read_text())[args.table]
    failures = []

    if args.table in {"table1_pascal_voc", "table4_generalized_voc"}:
        actual_cells = actual["cells"]
        keys = sorted(expected)
    else:
        actual_cells = {
            key.replace("shot", ""): value for key, value in actual["cells"].items()
        }
        keys = sorted(expected, key=int)

    for key in keys:
        if key not in actual_cells:
            if args.allow_missing:
                print(f"{key:16s} SKIP (not reproduced in this run)")
                continue
            failures.append(f"missing cell {key}")
            continue
        got = actual_cells[key]["selected"]
        if args.table == "table4_generalized_voc":
            target = expected[key]
            fields = ("AP50", "bAP50", "nAP50")
        else:
            target = expected[key]["selected"]
            fields = ("nAP50",) if args.table == "table1_pascal_voc" else ("nAP", "nAP50", "nAP75")
        for field in fields:
            delta = float(got[field]) - float(target[field])
            status = "OK" if abs(delta) <= args.tolerance else "FAIL"
            print(
                f"{key:16s} {field:6s} actual={got[field]:8.4f} "
                f"paper={target[field]:8.4f} delta={delta:+7.4f} {status}"
            )
            if status == "FAIL":
                failures.append(f"{key} {field} delta={delta:+.4f}")

    if failures:
        print("\nComparison: FAIL")
        for item in failures:
            print(" - " + item)
        return 2
    print("\nComparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

