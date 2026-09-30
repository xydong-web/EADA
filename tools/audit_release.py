import ast
import json
from pathlib import Path
import re
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = tuple(
    "".join(parts)
    for parts in (
        ("EC", "EA"),
        ("RI", "SF"),
        ("DA", "SR"),
        ("SP", "SD"),
        ("MF", "DC"),
        ("EA", "EC"),
    )
)
ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".sh"}
ALLOWED_SPECIAL_FILES = {"LICENSE", ".gitignore"}


def violations(relative, text):
    failures = []
    for token in FORBIDDEN:
        if token.casefold() in (relative + "\n" + text).casefold():
            failures.append(relative + ": forbidden method token " + token)
    if re.search(r"/(?:nfs|home|scratch|mnt)/", text):
        failures.append(relative + ": absolute machine path")
    return failures


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = config.pop("_BASE_", None)
    result = load_config(path.parent / base) if base else {}
    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict):
                merge(destination.setdefault(key, {}), value)
            else:
                destination[key] = value
    merge(result, config)
    return result


def audit(root=ROOT):
    failures = []
    files = []
    required = {
        "README.md",
        "INSTALL.md",
        "DATA.md",
        "MODEL_ZOO.md",
        "REPRODUCE.md",
        "environment.yml",
        "LICENSE",
        "ATTRIBUTION.md",
        "requirements.txt",
        "train_net.py",
        "eada/model.py",
        "eada/saia.py",
        "eada/losses.py",
        "eada/spdi.py",
        "scripts/train.sh",
        "scripts/eval.sh",
        "scripts/train_fsis.sh",
        "scripts/run_voc_cell.sh",
        "scripts/run_coco_cell.sh",
        "scripts/reviewer_quick_reproduce.sh",
        "scripts/reproduce_selected_voc.sh",
        "scripts/reproduce_selected_coco.sh",
        "scripts/reproduce_table1_voc.sh",
        "scripts/reproduce_table2_coco.sh",
        "scripts/reproduce_table3_fsis.sh",
        "scripts/reproduce_table5_ablation.sh",
        "scripts/reproduce_table6_spdi.sh",
        "scripts/reproduce_table7_diagnostics.sh",
        "scripts/reproduce_table8_sensitivity.sh",
        "scripts/reproduce_table9_efficiency.sh",
        "scripts/verify_release.sh",
        "tools/check_environment.py",
        "tools/check_data.py",
        "tools/check_weights.py",
        "tools/verify_auxiliary.py",
        "tools/aggregate_runs.py",
        "tools/compare_expected.py",
        "expected/paper_results.json",
        "tools/merge_fsis_checkpoint.py",
        "tools/verify_fsis_freeze.py",
        "tests/test_release.py",
    }
    for name in required:
        if not (root / name).is_file():
            failures.append(name + ": required file missing")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            # Runtime bytecode is never included in the release archive. Ignore
            # local interpreter caches so running the test suite does not make
            # the source-tree audit fail afterwards.
            continue
        if path.is_symlink():
            failures.append(relative + ": symlink forbidden")
            continue
        if path.is_dir():
            if path.name in {"datasets", "checkpoints", "logs", "outputs", ".git"}:
                failures.append(relative + ": excluded directory")
            continue
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in ALLOWED_SPECIAL_FILES:
            failures.append(relative + ": unexpected artifact")
            continue
        text = path.read_text(encoding="utf-8")
        failures.extend(violations(relative, text))
        if path.suffix == ".py":
            ast.parse(text, filename=relative)
            compile(text, relative, "exec")
        if path.suffix == ".yaml":
            config = load_config(path)
            assert config["MODEL"]["RESNETS"]["DEPTH"] == 101
            assert config["MODEL"]["RESNETS"]["OUT_FEATURES"] == ["res4"]
            predictor_mode = config["MODEL"]["EADA"]["ROI_PREDICTOR_MODE"]
            assert predictor_mode in {
                "standard", "residual_trainable", "residual_frozen"
            }
            if "shot" in relative:
                assert config["MODEL"]["EADA"]["SAIA_ENABLED"]
                assert config["MODEL"]["EADA"]["DC_ENABLED"]
                assert config["MODEL"]["EADA"]["SAIA_ALPHA"] == (
                    0.2 if "voc" in relative else 0.3
                )
                assert config["DATASETS"]["TRAIN"] and config["DATASETS"]["TEST"]
                assert config["DATASETS"]["EADA_TWO_STREAM"]
                assert config["SOLVER"]["IMS_PER_BATCH"] == 4
                assert config["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"] == 512
                if relative.startswith("configs/voc/split"):
                    assert predictor_mode == "residual_frozen"
                if relative.startswith("configs/coco/"):
                    assert predictor_mode == "standard"
            if "/fsis/" in "/" + relative:
                assert config["MODEL"]["MASK_ON"]
                if path.name == "base_mask.yaml":
                    assert not config["MODEL"]["EADA"]["FSIS_MASK_ONLY"]
                else:
                    assert config["MODEL"]["EADA"]["FSIS_MASK_ONLY"]
                assert config["MODEL"]["ROI_MASK_HEAD"]["CLS_AGNOSTIC_MASK"]
                assert config["MODEL"]["ROI_MASK_HEAD"]["NUM_CONV"] == 4
                assert config["MODEL"]["ROI_MASK_HEAD"]["CONV_DIM"] == 256
                assert config["MODEL"]["ROI_MASK_HEAD"]["POOLER_RESOLUTION"] == 14
        files.append(relative)
    for seed in (0, 1, 2):
        voc_files = list((root / f"splits/voc/seed{seed}").glob("box_*shot_*_train.txt"))
        coco_files = list((root / f"splits/coco/seed{seed}").glob("full_box_*shot_*_trainval.json"))
        if len(voc_files) != 100:
            failures.append(f"splits/voc/seed{seed}: expected 100 support files, got {len(voc_files)}")
        if len(coco_files) != 480:
            failures.append(f"splits/coco/seed{seed}: expected 480 support files, got {len(coco_files)}")
    if failures:
        raise AssertionError("\n".join(failures))
    return files


if __name__ == "__main__":
    files = audit()
    print(json.dumps({"status": "PASS", "files": len(files), "python_files": sum(name.endswith('.py') for name in files), "configs": sum(name.endswith('.yaml') for name in files)}, indent=2))
