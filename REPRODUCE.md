# Reproducing the paper experiments

This document gives the complete command sequence. The repository README
contains the same main workflow in condensed form.

## 1. Verify installation and data

```bash
conda activate eada
export DETECTRON2_DATASETS=/absolute/path/to/eada_datasets
bash scripts/setup_splits.sh
python tools/check_environment.py
python tools/check_data.py
```

## 2. Base training

### Recommended reviewer path: exact initialization bundle

If the companion reviewer initialization bundle is available, extract it into
the repository root and verify the file sizes in `MODEL_ZOO.md`. Then skip base
training and start directly from Section 3. This uses the same expanded
base/model initialization states as the reported paper runs.

```bash
python tools/check_weights.py
```

### Full from-scratch path

PASCAL VOC has three independent base detectors, one per split:

```bash
NUM_GPUS=4 bash scripts/train_voc_base.sh 1
NUM_GPUS=4 bash scripts/train_voc_base.sh 2
NUM_GPUS=4 bash scripts/train_voc_base.sh 3
```

COCO uses one base detector:

```bash
NUM_GPUS=4 bash scripts/train_coco_base.sh
```

Each script trains the base detector and then expands the classification and
box-regression heads to the full base+novel label space.

This from-scratch route verifies the complete training pipeline. Because the
historical novel-class head surgery used random initialization, a regenerated
expanded checkpoint is not expected to be bitwise identical to the archived
paper initialization. For strict metric comparison, use the companion bundle.

## 3. A single VOC paper cell

Example: Split 1, 1 shot, seed 0:

```bash
NUM_GPUS=4 bash scripts/run_voc_cell.sh 1 1 0
```

With the exact initialization bundle already extracted, the same end-to-end
reviewer check including all preflight checks and paper-value comparison is:

```bash
NUM_GPUS=4 bash scripts/reviewer_quick_reproduce.sh
```

The script performs few-shot adaptation with SAIA+DC, then evaluates the frozen
checkpoint with pre-NMS SPDI. Output is written to:

```text
outputs/voc/split1/1shot/seed0/
├── train/
└── eval/
```

## 4. Table 1 and Table 4: complete VOC matrix

If the goal is to reproduce the exact table cells using the already recorded
selected seeds, run the shorter reviewer path:

```bash
NUM_GPUS=4 bash scripts/reproduce_selected_voc.sh
```

To independently rerun the three-seed selection procedure, use the full
matrix below.

Run three seeds for all 3 splits and all 1/2/3/5/10-shot settings:

```bash
NUM_GPUS=4 bash scripts/reproduce_table1_voc.sh
```

After all jobs finish:

```bash
python tools/aggregate_runs.py \
  --dataset voc \
  --root outputs/voc \
  --output outputs/voc/summary.json

python tools/compare_expected.py \
  --results outputs/voc/summary.json \
  --table table1_pascal_voc
```

For each split/shot cell the paper reports the seed with the highest novel
AP50; ties are broken by novel AP. All AP/AP50/AP75 values in that table row are
taken from the same selected run. The same evaluations also produce overall
AP50 used for generalized FSOD in Table 4, so Table 4 requires no extra model
training.

Verify the Table-4 generalized metrics from the same aggregate file:

```bash
python tools/compare_expected.py \
  --results outputs/voc/summary.json \
  --table table4_generalized_voc
```

## 5. Table 2: complete COCO matrix

Single-cell example:

```bash
NUM_GPUS=4 bash scripts/run_coco_cell.sh 10 2
```

Selected-seed reviewer path:

```bash
NUM_GPUS=4 bash scripts/reproduce_selected_coco.sh
```

Full 1/2/3/5/10/30-shot three-seed matrix:

```bash
NUM_GPUS=4 bash scripts/reproduce_table2_coco.sh
```

Aggregate and compare:

```bash
python tools/aggregate_runs.py \
  --dataset coco \
  --root outputs/coco \
  --output outputs/coco/summary.json

python tools/compare_expected.py \
  --results outputs/coco/summary.json \
  --table table2_ms_coco
```

The selected COCO seeds in the paper are 0, 2, 2, 2, 2, and 1 for
1/2/3/5/10/30 shots respectively.

## 6. Table 3: FSIS extension

First train the class-agnostic base mask head:

```bash
NUM_GPUS=4 bash scripts/train_coco_base_mask.sh
```

If `weights/coco_base_mask_init.pth` is present from the companion bundle, the
Table-3 runner uses it automatically and the base-mask training command can be
skipped.

Then reproduce all six selected detector cells and mask-only adaptation:

```bash
NUM_GPUS=4 bash scripts/reproduce_table3_fsis.sh
```

The detector is frozen during mask adaptation. Only parameters under
`roi_heads.mask_head` are trainable. `tools/verify_fsis_freeze.py` is executed
by the script after every training run.

## 7. Table 5: component ablation

The component study is run on VOC Split 1, seed 0, at 1/3/5 shots:

```bash
NUM_GPUS=4 bash scripts/reproduce_table5_ablation.sh
```

The six rows correspond to:

```text
Baseline       SAIA off, DC off, SPDI off
+DC            SAIA off, DC on,  SPDI off
+SAIA          SAIA on,  DC off, SPDI off
+DC+SAIA       SAIA on,  DC on,  SPDI off
+SPDI          SAIA off, DC off, SPDI on
Full EADA      SAIA on,  DC on,  SPDI on
```

The script reuses the Baseline and DC+SAIA checkpoints for their corresponding
SPDI rows so that the semantic integration comparison is inference-only.

## 8. Table 6: SPDI controls

Run the placement/fusion controls on the frozen Split-1 checkpoints:

```bash
bash scripts/reproduce_table6_spdi.sh
```

The final method uses `TEST.SPDI_FUSION_MODE absolute`. The normalized control
preserves the detector novel-class probability mass and only changes the
conditional distribution within the novel subset. The post-NMS control is
included only as a placement diagnostic; it cannot recover hypotheses already
removed by detector thresholding/NMS.

## 9. Tables 7--9

Mechanism and efficiency analyses use dedicated scripts:

```bash
bash scripts/reproduce_table7_diagnostics.sh
bash scripts/reproduce_table8_sensitivity.sh
bash scripts/reproduce_table9_efficiency.sh
```

These analyses are intentionally separated from the main benchmark runners so
they do not change the training checkpoints used for Tables 1--4.

Each auxiliary runner automatically compares its generated JSON outputs with
`expected/paper_results.json` through `tools/verify_auxiliary.py`. Efficiency
results are reported without a hard failure by default because Table 9 is
hardware-sensitive.

## 10. Expected values and numerical tolerance

All reported target values and selected seed IDs are stored in
`expected/paper_results.json`.

Exact floating-point equality is not expected across GPU architectures,
compiler versions, or CUDA kernels. The comparison utility therefore defaults
to a 0.5 AP-point tolerance. This tolerance is a reproducibility check, not a
new evaluation rule; paper values remain the values recorded in the manuscript.

## 11. Reproduction levels

For reviewer convenience, three levels are supported:

1. **Code-path smoke test** -- run unit tests and build the models; no dataset or
   long training is needed.
2. **Single-cell reproduction** -- train one VOC or COCO cell end-to-end and
   compare it with `expected/paper_results.json`.
3. **Full-paper reproduction** -- train all seeds/cells and regenerate the
   selected rows for Tables 1--4 plus the requested ablations.

The full matrix is computationally expensive. Reviewers who only need to
verify the proposed mechanisms can use a single Split-1 VOC cell together with
the Table 5/6 scripts.

