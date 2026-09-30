#!/usr/bin/env python3
"""Compare reproduced auxiliary tables with the paper targets.

Tables 3 and 5--8 use a default absolute tolerance of 0.5 AP/percentage
points. Table 9 is hardware-sensitive and is report-only unless --strict is
requested.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "expected/paper_results.json").read_text())


def load(path):
    path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def bbox(path):
    payload = load(path)
    if "bbox" in payload:
        return payload["bbox"]
    return payload["metrics"]["bbox"]


def compare(label, actual, target, tolerance, failures):
    delta = float(actual) - float(target)
    ok = abs(delta) <= tolerance
    print(
        f"{label:48s} actual={float(actual):9.4f} "
        f"paper={float(target):9.4f} delta={delta:+8.4f} "
        f"{'OK' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(label)


def table3(tolerance, failures):
    expected = EXPECTED["table3_fsis"]
    for shot in (1, 2, 3, 5, 10, 30):
        seed = expected[f"{shot}shot"]["selected_seed"]
        result = load(
            f"outputs/fsis/{shot}shot/seed{seed}/eval/evaluation_results.json"
        )
        for task in ("bbox", "segm"):
            got = result[task]
            target = expected[f"{shot}shot"][task]
            for metric in ("nAP", "nAP50", "nAP75"):
                compare(
                    f"Table3 {shot}shot {task} {metric}",
                    got[metric], target[metric], tolerance, failures,
                )


def table5(tolerance, failures):
    expected = EXPECTED["table5_component_ablation"]
    shots = expected["shots"]
    rows = {
        "Baseline": "baseline",
        "DC": "dc",
        "SAIA": "saia",
        "DC+SAIA": "dc_saia",
        "SPDI": "spdi",
        "Full EADA": "full_eada",
    }
    for label, directory in rows.items():
        for index, shot in enumerate(shots):
            got = bbox(
                f"outputs/ablation/table5/{shot}shot/{directory}/eval/"
                "evaluation_results.json"
            )["nAP50"]
            target = expected["novel_AP50"][label][index]
            compare(
                f"Table5 {label} {shot}shot nAP50",
                got, target, tolerance, failures,
            )


def table6(tolerance, failures):
    expected = EXPECTED["table6_spdi_design"]
    placement = {
        "Detector-only": "detector_only",
        "Post-NMS SPDI": "post_nms_spdi",
        "Pre-NMS SPDI": "pre_nms_spdi",
    }
    for label, directory in placement.items():
        got = bbox(
            f"outputs/ablation/table6/1shot/{directory}/evaluation_results.json"
        )
        for metric in ("nAP", "nAP50", "nAP75"):
            compare(
                f"Table6 placement {label} {metric}",
                got[metric], expected["placement_1shot"][label][metric],
                tolerance, failures,
            )
    fusion = {
        "Detector-only": "detector_only",
        "Normalized pre-NMS": "normalized_pre_nms",
        "Absolute-score pre-NMS": "absolute_pre_nms",
    }
    for label, directory in fusion.items():
        got = bbox(
            f"outputs/ablation/table6/10shot/{directory}/evaluation_results.json"
        )
        for metric in ("nAP", "nAP50", "nAP75"):
            compare(
                f"Table6 fusion {label} {metric}",
                got[metric], expected["fusion_10shot"][label][metric],
                tolerance, failures,
            )


def table7(tolerance, failures):
    expected = EXPECTED["table7_mechanism_diagnostics"]
    control = load("outputs/diagnostics/table7/saia_control.json")
    proposed = load("outputs/diagnostics/table7/saia_proposed.json")
    saia_fields = {
        "Recall@100_IoU0.50": "recall_at_100_iou50_percent",
        "Recall@100_IoU0.75": "recall_at_100_iou75_percent",
        "Mean_best_IoU": "mean_best_iou_percent",
    }
    for label, field in saia_fields.items():
        targets = expected["SAIA"][label]
        compare(f"Table7 SAIA control {label}", control[field], targets[0], tolerance, failures)
        compare(f"Table7 SAIA proposed {label}", proposed[field], targets[1], tolerance, failures)

    gradients = load("outputs/diagnostics/table7/dc_gradient.json")
    dc_fields = {
        "Absent_class_gradient_x1e-3": "absent_class_gradient_x1e3",
        "Supported_class_gradient_x1e-3": "supported_class_gradient_x1e3",
    }
    for label, field in dc_fields.items():
        targets = expected["DC"][label]
        compare(f"Table7 DC control {label}", gradients[field]["control_ce"], targets[0], tolerance, failures)
        compare(f"Table7 DC proposed {label}", gradients[field]["dc"], targets[1], tolerance, failures)


def table8(tolerance, failures):
    expected = EXPECTED["table8_sensitivity"]
    specs = [
        ("SAIA_alpha", "saia_alpha", True),
        ("SPDI_lambda_novel", "lambda", False),
        ("SPDI_temperature", "temperature", False),
        ("SPDI_topk", "topk", False),
    ]
    for key, directory, has_eval_subdir in specs:
        values = expected[key]["values"]
        targets = expected[key]["novel_AP50"]
        for value, target in zip(values, targets):
            suffix = "/eval" if has_eval_subdir else ""
            got = bbox(
                f"outputs/sensitivity/{directory}/{value}{suffix}/"
                "evaluation_results.json"
            )["nAP50"]
            compare(f"Table8 {key}={value} nAP50", got, target, tolerance, failures)


def table9(tolerance, failures, strict):
    expected = EXPECTED["table9_efficiency_rtx4070ti"]
    specs = {
        "Baseline": "outputs/efficiency/baseline.json",
        "DC+SAIA": "outputs/efficiency/dc_saia.json",
        "Full EADA": "outputs/efficiency/full_eada.json",
    }
    fields = {
        "detector_M": lambda x: x["detector_parameters"] / 1e6,
        "clip_M": lambda x: x["clip_parameters"] / 1e6,
        "GFLOPs": lambda x: x["upper_bound_gflops_per_image"],
        "infer_GiB": lambda x: x["inference_peak_memory_gib"],
        "latency_ms": lambda x: x["latency_ms_per_image"],
        "FPS": lambda x: x["fps"],
    }
    local_failures = []
    for label, path in specs.items():
        got = load(path)
        print(f"\n{label} on {got.get('gpu', 'unknown GPU')}")
        for field, getter in fields.items():
            compare(
                f"Table9 {label} {field}", getter(got), expected[label][field],
                tolerance, local_failures,
            )
    if strict:
        failures.extend(local_failures)
    elif local_failures:
        print("\nTable 9 is hardware-sensitive; differences above are report-only. "
              "Use --strict to make them fail verification.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        choices=("table3", "table5", "table6", "table7", "table8", "table9"),
        required=True,
    )
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    failures = []
    globals()[args.table](args.tolerance, failures, args.strict) if args.table == "table9" else globals()[args.table](args.tolerance, failures)
    if failures:
        print("\nVerification: FAIL")
        for item in failures:
            print(" - " + item)
        return 2
    print("\nVerification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

