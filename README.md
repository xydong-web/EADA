# EADA: Evidence-Aligned Decoupled Adaptation

Official reviewer-reproduction package for the paper **"Evidence-Aligned Decoupled Adaptation for Few-Shot Object Detection"**.

This repository is designed for paper review rather than only as a minimal method demonstration. It contains the complete public EADA implementation, exact seed-0/1/2 few-shot support splits used by the reported experiments, all VOC/COCO training configurations, checkpoint surgery utilities, FSOD/FSIS evaluation code, paper-result targets, and scripts that reproduce the main benchmark and ablation tables.

> **Reviewer entry point.** If you want to verify one result before launching the full experiment matrix, follow **Reviewer Quick Reproduction** below. A single PASCAL VOC Split-1 cell exercises the full SAIA + DC training path and the frozen pre-NMS SPDI inference path.

> **Strict paper initialization.** The source archive is accompanied by an
> optional initialization-checkpoint bundle. Extracting that bundle into this
> repository creates `weights/voc_split{1,2,3}_all.pth`, `weights/coco_all.pth`,
> and `weights/coco_base_mask_init.pth`. These are the exact expanded base/model
> initialization checkpoints used by the reported paper runs. They are the
> recommended starting point for reviewer reproduction because they remove a
> historical random classifier-surgery initialization from the comparison.

If both reviewer archives are provided, prepare them in one directory with:

```bash
unzip EADA_Reviewer_Reproduction_Code.zip
unzip EADA_Reviewer_Initialization_Weights.zip
cd EADA
python tools/check_weights.py
```

Both archives use the same top-level `EADA/` directory, so the second command
only adds the `weights/` initialization files.

---

## 1. What is EADA?

EADA separates the evidence used for few-shot **adaptation** from the evidence used for final semantic **decision integration**.

The public method vocabulary in this release exactly follows the paper:

- **SAIA — Spatially Adaptive Instance Aggregation.** A four-layer deformable-attention encoder refines the active ResNet-C4 spatial representation before the RPN.
- **DC — Image-Conditioned Decoupled Classification.** During few-shot adaptation, foreground RoIs retain the full classifier, while a background RoI only backpropagates category-specific classification gradients through foreground classes annotated in its current image plus the background coordinate.
- **SPDI — Semantic-Prior Decision Integration.** During inference, frozen CLIP ViT-B/16 proposal-crop evidence is fused into **novel-category** detector scores before confidence filtering and class-wise NMS.
- **EADA — SAIA + DC + SPDI.**

The detector remains a standard ResNet-101-C4 two-stage detector. SPDI never participates in detector optimization.

### 1.1 Exact implementation placement

The execution path is:

```text
input image
   |
ResNet-101-C4
   |
active res4 feature
   |
  SAIA                         training + inference
   |
   +------------------> RPN --------------------+
   |                                            |
   +------------------> RoI head <--------------+
                           |
                 classifier / box regressor
                           |
                  DC classification loss        training only
                           |
                   pre-NMS detector scores
                           |
                  frozen CLIP proposal crops
                           |
                          SPDI                   inference only
                           |
                  threshold + class-wise NMS
                           |
                    final detections
```

For the optional FSIS extension, a class-agnostic Mask R-CNN head is attached after EADA detector adaptation. All detector parameters are frozen and only the mask head is optimized.

---

## 2. Reviewer Quick Reproduction

The fastest end-to-end verification is PASCAL VOC Split 1, 1-shot, seed 0.

### Step 1 — install the environment

`environment.yml` creates the Python/PyTorch base environment. SAIA also
requires a CUDA-enabled MMCV 1.7.2 build, and this release uses the legacy
Detectron2 0.3 API. Install those two compiled dependencies before running the
self-check:

```bash
conda env create -f environment.yml
conda activate eada

git clone --branch v1.7.2 --depth 1 https://github.com/open-mmlab/mmcv.git third_party/mmcv
MMCV_WITH_OPS=1 python -m pip install -v -e third_party/mmcv

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.3'
python -m pip install -r requirements.txt
```

MMCV must include compiled deformable-attention operators. Detectron2 and MMCV
must be built against the same PyTorch/CUDA toolchain visible at runtime.
Detailed compiler notes and an alternative local-clone installation are in
[INSTALL.md](INSTALL.md).

Verify the environment:

```bash
python tools/check_environment.py
```

### Step 2 — prepare VOC and install the bundled support splits

```bash
export DETECTRON2_DATASETS=/absolute/path/to/eada_datasets
bash scripts/setup_splits.sh
python tools/check_data.py --dataset voc
```

For this VOC-only quick test, the expected raw data are:

```text
$DETECTRON2_DATASETS/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/Main/
│   └── JPEGImages/
├── VOC2012/
│   ├── Annotations/
│   ├── ImageSets/Main/
│   └── JPEGImages/
└── vocsplit/
    ├── seed0/
    ├── seed1/
    └── seed2/
```

The support files are already included under `splits/voc/`; `scripts/setup_splits.sh` installs them into your dataset root.

### Step 3 — choose the initialization path

For the shortest and strictest reviewer reproduction, extract the companion
initialization bundle so that this file exists:

```text
weights/voc_split1_all.pth
```

Verify it against [MODEL_ZOO.md](MODEL_ZOO.md), then skip directly to Step 5.
This is the path used for paper-cell reproduction.

```bash
python tools/check_weights.py --detection-only
```

To reproduce the complete base-training pipeline from ImageNet instead, place
the Detectron-style ResNet-101 ImageNet checkpoint at:

Place the Detectron-style ResNet-101 ImageNet checkpoint at:

```text
weights/R-101.pkl
```

The exact file used by the validated experiments has:

```text
size    178431595 bytes
```

See [MODEL_ZOO.md](MODEL_ZOO.md).

### Step 4 — optional from-scratch base training and head expansion

```bash
NUM_GPUS=4 bash scripts/train_voc_base.sh 1
```

This performs the standard transfer-based FSOD initialization sequence:

1. train the base detector on the 15 base categories;
2. preserve all shared detector parameters;
3. expand classifier and box-regression tensors to the joint 20-class space;
4. retain base-class weights and randomly initialize novel-class rows.

The expanded checkpoint is written to:

```text
weights/voc_split1_all.pth
```

The from-scratch path is provided to reproduce the full training procedure.
For numerical comparison against the exact paper cells, the companion expanded
checkpoint is preferred: the historical classifier-surgery step initialized
novel rows randomly, so a newly generated expanded checkpoint is not expected
to be bitwise identical even when the base detector is retrained correctly.

### Step 5 — reproduce Split-1 / 1-shot / seed-0 EADA

```bash
NUM_GPUS=4 bash scripts/run_voc_cell.sh 1 1 0
```

After the environment, dataset root, support splits, and companion
initialization bundle are prepared, Steps 1/2/3/5/6 can also be executed as a
single reviewer command:

```bash
NUM_GPUS=4 bash scripts/reviewer_quick_reproduce.sh
```

The script checks the environment, validates the VOC dataset and initialization
file sizes, trains the Split-1 1-shot seed-0 detector, runs SPDI
evaluation, and compares the completed cell with the paper target.

The script runs:

```text
few-shot adaptation:
  support stream (4 images) + base stream (4 images)
  -> SAIA + DC
  -> detector checkpoint

evaluation:
  frozen EADA detector
  -> top-100 pre-NMS proposals
  -> frozen CLIP ViT-B/16
  -> SPDI absolute novel-score fusion
  -> threshold + NMS
```

The final metric file is:

```text
outputs/voc/split1/1shot/seed0/eval/evaluation_results.json
```

The paper-selected seed-0 target for this cell is approximately:

```text
novel AP     37.06
novel AP50   68.27
novel AP75   36.99
base AP50    78.05
overall AP50 75.60
```

Exact stored targets are in `expected/paper_results.json`.

### Step 6 — compare reproduced values

After one or more seeds have completed:

```bash
python tools/aggregate_runs.py   --dataset voc   --root outputs/voc   --output outputs/voc/summary.json

python tools/compare_expected.py   --results outputs/voc/summary.json   --table table1_pascal_voc
```

The default comparison tolerance is 0.5 AP point because CUDA kernels, GPU architecture, and compiled operator versions can introduce small numerical differences.

---

## 3. Repository Layout

```text
EADA/
├── README.md
├── INSTALL.md
├── DATA.md
├── MODEL_ZOO.md
├── REPRODUCE.md
├── environment.yml
├── requirements.txt
├── LICENSE
├── ATTRIBUTION.md
├── PROVENANCE.md
├── VERIFICATION.md
│
├── eada/
│   ├── config.py
│   ├── model.py                 # EADA detector and RoI head
│   ├── saia.py                  # Spatially Adaptive Instance Aggregation
│   ├── losses.py                # DC classification objective
│   ├── spdi.py                  # pre-/post-NMS semantic integration controls
│   ├── data/
│   │   ├── builtin.py
│   │   ├── builtin_meta.py
│   │   ├── meta_voc.py
│   │   └── meta_coco.py
│   └── evaluation/
│       ├── pascal_voc_evaluation.py
│       └── coco_evaluation.py
│
├── configs/
│   ├── Base-EADA.yaml
│   ├── voc/
│   │   ├── base{1,2,3}.yaml
│   │   └── split{1,2,3}_{1,2,3,5,10}shot.yaml
│   └── coco/
│       ├── base.yaml
│       ├── {1,2,3,5,10,30}shot.yaml
│       └── fsis/
│
├── splits/
│   ├── voc/seed{0,1,2}/
│   └── coco/seed{0,1,2}/
│
├── expected/
│   └── paper_results.json
│
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   ├── train_fsis.sh
│   ├── setup_splits.sh
│   ├── train_voc_base.sh
│   ├── train_coco_base.sh
│   ├── run_voc_cell.sh
│   ├── run_coco_cell.sh
│   ├── reviewer_quick_reproduce.sh
│   ├── reproduce_selected_voc.sh
│   ├── reproduce_selected_coco.sh
│   ├── reproduce_table1_voc.sh
│   ├── reproduce_table2_coco.sh
│   ├── reproduce_table3_fsis.sh
│   ├── reproduce_table5_ablation.sh
│   ├── reproduce_table6_spdi.sh
│   ├── reproduce_table7_diagnostics.sh
│   ├── reproduce_table8_sensitivity.sh
│   ├── reproduce_table9_efficiency.sh
│   └── verify_release.sh
│
├── tools/
│   ├── expand_checkpoint.py
│   ├── merge_fsis_checkpoint.py
│   ├── verify_fsis_freeze.py
│   ├── check_environment.py
│   ├── check_data.py
│   ├── aggregate_runs.py
│   ├── compare_expected.py
│   ├── saia_proposal_recall.py
│   ├── dc_gradient_diagnostic.py
│   ├── profile_efficiency.py
│   ├── verify_auxiliary.py
│   └── audit_release.py
│
└── tests/
    └── test_release.py
```

Raw datasets, pretrained weights, and trained checkpoints are deliberately excluded from the source archive.
The exact expanded base/model initializations are distributed separately so the
source archive remains manageable while reviewers can still start from the
same paper initialization state.

---

## 4. Verified Environment

The final experiment workspace used:

| Component | Version |
| --- | --- |
| OS | Linux |
| Python | 3.9.23 |
| PyTorch | 2.2.2 + CUDA 12.1 |
| torchvision | 0.17.2 + CUDA 12.1 |
| Detectron2 | 0.3 legacy API |
| MMCV | 1.7.2 with compiled ops |
| MMDetection | 2.28.2 |
| OpenCLIP | 2.24.0 |
| NumPy | 1.26.4 |

Run:

```bash
python tools/check_environment.py
```

The checker verifies version compatibility and imports the MMCV multi-scale deformable-attention operator required by SAIA.

Detailed installation notes are in [INSTALL.md](INSTALL.md).

---

## 5. Dataset Protocol

### 5.1 PASCAL VOC

We use the standard three-split FSOD protocol:

- VOC 2007 + VOC 2012 `trainval` for base/few-shot training;
- VOC 2007 `test` for evaluation;
- 15 base + 5 novel classes per split;
- shots: 1, 2, 3, 5, 10.

Paper experiments use the fixed seed set:

```text
seed 0
seed 1
seed 2
```

The exact support files used by those seeds are bundled in this release.

### 5.2 MS COCO

We use the standard 60-base / 20-novel partition in which the 20 categories shared with PASCAL VOC are novel.

Shots:

```text
1, 2, 3, 5, 10, 30
```

The benchmark uses the COCO 2014 `trainvalno5k` / `5k` split. See [DATA.md](DATA.md) for the required directory tree.

If a normal COCO download only provides `train2014/` and `val2014/`, create the
combined `trainval2014/` image view expected by this code with:

```bash
bash scripts/prepare_coco_images.sh
```

### 5.3 Install the exact paper support splits

```bash
export DETECTRON2_DATASETS=/absolute/path/to/eada_datasets
bash scripts/setup_splits.sh
```

By default the script copies the packaged seed directories into the dataset root so that the
dataset remains valid even if the source archive is moved. To use symbolic links instead:

```bash
COPY_SPLITS=0 bash scripts/setup_splits.sh
```

### 5.4 Validate all data

```bash
python tools/check_data.py
```

For COCO this also checks that `trainvalno5k.json` and `5k.json` are present.
For a single-dataset reviewer check, use `--dataset voc` or `--dataset coco`.

---

## 6. Pretrained Models

### 6.1 ImageNet ResNet-101

Base training starts from `weights/R-101.pkl`.

Validated file size:

```text
178431595 bytes
```

### 6.2 CLIP ViT-B/16

SPDI uses frozen OpenAI CLIP ViT-B/16.

Online mode: leave `TEST.SPDI_CHECKPOINT` empty and OpenCLIP will use its `openai` pretrained identifier.

Offline mode:

```bash
export EADA_CLIP_CHECKPOINT=/absolute/path/to/ViT-B-16.pt
```

The checkpoint used in the validated experiments has an approximate size of:

```text
approximately 335 MiB
```

SPDI never updates CLIP parameters.

### 6.3 Exact reviewer initialization checkpoints

The optional companion bundle should be extracted directly into the repository
root. It provides:

```text
weights/voc_split1_all.pth
weights/voc_split2_all.pth
weights/voc_split3_all.pth
weights/coco_all.pth
weights/coco_base_mask_init.pth
```

These checkpoints contain no final EADA few-shot results. They are only the
initialization states needed to reproduce the reported adaptation and FSIS
experiments without rerunning base pretraining or changing the historical
novel-class initialization. Expected file sizes are listed in
[MODEL_ZOO.md](MODEL_ZOO.md).

Verify the complete companion bundle with:

```bash
python tools/check_weights.py
```

---

## 7. Exact Training Protocol

### 7.1 Base training

VOC base training is independent for each split.

| Dataset | GPUs | IMS_PER_BATCH | LR | Max iter | LR steps | Warmup |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| VOC split 1/2/3 | 4 | 32 | 0.01 | 15000 | 10000, 13300 | 100 |
| COCO | 4 | 16 | 0.01 | 220000 | 170000, 200000 | 1000 |

Base training commands:

```bash
NUM_GPUS=4 bash scripts/train_voc_base.sh 1
NUM_GPUS=4 bash scripts/train_voc_base.sh 2
NUM_GPUS=4 bash scripts/train_voc_base.sh 3
NUM_GPUS=4 bash scripts/train_coco_base.sh
```

### 7.2 Few-shot adaptation

Every paper few-shot training run uses a two-stream loader:

```text
support stream: SOLVER.IMS_PER_BATCH = 4
base stream:    SOLVER.IMS_PER_BATCH = 4
combined model-forward batch per optimizer step = 8 images
```

Both streams are concatenated before a single optimizer step.

Common optimizer settings:

```text
optimizer        SGD
momentum         0.9
weight decay     5e-5
RoIs / image     512
classifier drop  0.8
RPN grad scale   0.0
```

VOC:

| Shots | LR | Max iter | LR decay | SAIA α | RoI grad scale |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0025 | 1600 | 1200 | 0.2 | 0.001 |
| 2 | 0.0025 | 1600 | 1200 | 0.2 | 0.001 |
| 3 | 0.0025 | 2400 | 2000 | 0.2 | 0.001 |
| 5 | 0.0025 | 3000 | 2400 | 0.2 | 0.001 |
| 10 | 0.0025 | 3000 | 2400 | 0.2 | 0.001 |

COCO:

| Shots | LR | Max iter | LR decay | SAIA α | RoI grad scale |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.005 | 3200 | 2800 | 0.3 | 0.01 |
| 2 | 0.005 | 3600 | 3200 | 0.3 | 0.01 |
| 3 | 0.005 | 4000 | 3600 | 0.3 | 0.01 |
| 5 | 0.005 | 4400 | 4000 | 0.3 | 0.01 |
| 10 | 0.005 | 5600 | 4800 | 0.3 | 0.01 |
| 30 | 0.005 | 9600 | 8000 | 0.3 | 0.01 |

### 7.2.1 Reviewer table reproduction versus full seed-selection reproduction

Two modes are provided. To reproduce only the runs actually selected into the
paper tables, use:

```bash
NUM_GPUS=4 bash scripts/reproduce_selected_voc.sh
NUM_GPUS=4 bash scripts/reproduce_selected_coco.sh
```

These launch 15 VOC cells and 6 COCO cells using the selected seed IDs stored
in `expected/paper_results.json`.

To independently repeat the complete selection protocol, rerun all three seeds
for every cell:

```bash
NUM_GPUS=4 bash scripts/reproduce_table1_voc.sh
NUM_GPUS=4 bash scripts/reproduce_table2_coco.sh
```

These launch 45 VOC cells and 18 COCO cells. The aggregation utility then
selects the highest novel AP50 in each cell, using novel AP as the tiebreaker,
which is the selection rule used to generate the reported rows.

The Table 3 and Table 5--9 runners also call `tools/verify_auxiliary.py` at the
end. Tables 3 and 5--8 use the same default 0.5-point numerical tolerance as
the main benchmark comparator. Table 9 is report-only by default because
latency and memory depend on the exact GPU/driver; use `--strict` only when
profiling on the paper's RTX 4070 Ti environment.

### 7.3 Benchmark-specific RoI predictor protocol

The released configs preserve the head protocols used by the final checkpoints:

- **Base pretraining (VOC and COCO):** `MODEL.EADA.ROI_PREDICTOR_MODE residual_trainable`. A learned 2048→2048 residual mapping precedes a trainable linear classifier.
- **VOC few-shot adaptation:** `MODEL.EADA.ROI_PREDICTOR_MODE residual_frozen`. The base-trained residual mapping remains active, while the expanded linear classifier is frozen.
- **COCO few-shot adaptation:** `MODEL.EADA.ROI_PREDICTOR_MODE standard`. The ordinary trainable Fast R-CNN classifier is used and the base-stage residual mapping is not part of the forward path.

This distinction is encoded in the YAML configs; do not force one setting across both benchmarks.

### 7.4 SAIA implementation dimensions

The active Detectron2 ResNet-101-C4 `res4` tensor has 1024 channels. The checkpoint-compatible SAIA path is therefore:

```text
1024 -> 256 -> four deformable-attention layers -> 1024
```

Residual fusion is:

```text
x_SAIA = (1 - alpha) * x + alpha * SAIA(x)
```

with α = 0.2 on VOC and 0.3 on COCO.

### 7.5 DC

For a foreground RoI, all classifier logits remain active.

For a background RoI from image `I`, the gate keeps:

```text
annotated foreground classes in I + background
```

and multiplies all unsupported foreground logits by zero before softmax. Those logits consequently receive zero direct gradient through that background RoI.

### 7.6 SPDI

Final EADA uses:

```text
CLIP model             ViT-B/16, OpenAI pretrained, frozen
proposal budget        top 100 pre-NMS proposals
crop size              224 x 224
text templates         "a photo of a {}"
                       "a clean photo of a {}"
temperature            0.01
detector weight        lambda_N = 0.7
fusion placement       pre-NMS
fusion mode            absolute
affected coordinates   novel classes only
base/background        unchanged
boxes                   unchanged
```

For a novel class `c`:

```text
s'(c) = 0.7 * s_detector(c) + 0.3 * s_CLIP(c)
```

The score vector is then passed to ordinary detector thresholding and class-wise NMS.

---

## 8. Reproducing Table 1 — PASCAL VOC FSOD

### 8.1 One cell

```bash
NUM_GPUS=4 bash scripts/run_voc_cell.sh SPLIT SHOT SEED
```

Example:

```bash
NUM_GPUS=4 bash scripts/run_voc_cell.sh 2 5 0
```

### 8.2 Full paper matrix

This runs:

```text
3 splits x 5 shot settings x 3 seeds = 45 training/evaluation cells
```

```bash
NUM_GPUS=4 bash scripts/reproduce_table1_voc.sh
```

You can restrict the matrix:

```bash
EADA_VOC_SPLITS="1" EADA_VOC_SHOTS="1 3 5" EADA_SEEDS="0 1 2" NUM_GPUS=4 bash scripts/reproduce_table1_voc.sh
```

Aggregate:

```bash
python tools/aggregate_runs.py   --dataset voc   --root outputs/voc   --output outputs/voc/summary.json
```

Compare with paper values:

```bash
python tools/compare_expected.py   --results outputs/voc/summary.json   --table table1_pascal_voc
```

---

## 9. Reproducing Table 2 — MS COCO FSOD

### 9.1 One cell

```bash
NUM_GPUS=4 bash scripts/run_coco_cell.sh SHOT SEED
```

Example:

```bash
NUM_GPUS=4 bash scripts/run_coco_cell.sh 10 2
```

### 9.2 Full paper matrix

```text
6 shot settings x 3 seeds = 18 training/evaluation cells
```

```bash
NUM_GPUS=4 bash scripts/reproduce_table2_coco.sh
```

Restrict if necessary:

```bash
EADA_COCO_SHOTS="1 5 10" EADA_SEEDS="0 1 2" NUM_GPUS=4 bash scripts/reproduce_table2_coco.sh
```

Aggregate and compare:

```bash
python tools/aggregate_runs.py   --dataset coco   --root outputs/coco   --output outputs/coco/summary.json

python tools/compare_expected.py   --results outputs/coco/summary.json   --table table2_ms_coco
```

Paper-selected COCO seeds and target novel metrics:

| Shot | Seed | AP | AP50 | AP75 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 12.7 | 23.1 | 12.0 |
| 2 | 2 | 14.5 | 28.0 | 12.8 |
| 3 | 2 | 16.0 | 30.9 | 14.4 |
| 5 | 2 | 18.4 | 34.2 | 17.5 |
| 10 | 2 | 21.0 | 38.4 | 20.6 |
| 30 | 1 | 23.3 | 42.5 | 22.4 |

---

## 10. Reproducing Table 3 — Few-Shot Instance Segmentation

First reproduce the COCO base detector, then train a class-agnostic base mask head:

```bash
NUM_GPUS=4 bash scripts/train_coco_base.sh
NUM_GPUS=4 bash scripts/train_coco_base_mask.sh
```

Run all six FSIS cells:

```bash
NUM_GPUS=4 bash scripts/reproduce_table3_fsis.sh
```

The script:

1. resolves the paper-selected detector seed for each shot;
2. reproduces that EADA detector if it is missing;
3. merges the base class-agnostic mask head into the frozen EADA detector;
4. trains **only** `roi_heads.mask_head`;
5. verifies detector tensors did not change with `tools/verify_fsis_freeze.py`;
6. evaluates detection and segmentation with SPDI active before NMS.

Paper novel mask AP targets:

| Shot | Selected detector seed | Mask AP |
| ---: | ---: | ---: |
| 1 | 0 | 9.6 |
| 2 | 2 | 10.7 |
| 3 | 2 | 12.1 |
| 5 | 2 | 13.6 |
| 10 | 2 | 15.6 |
| 30 | 1 | 17.4 |

---

## 11. Reproducing Table 4 — Generalized FSOD

No extra training is required.

The VOC evaluation in Table 1 is performed over the joint base+novel output space and returns:

```text
AP / AP50 / AP75
bAP / bAP50 / bAP75
nAP / nAP50 / nAP75
```

Therefore Table 4 overall AP50 and the base-preservation analysis can be read from the same `evaluation_results.json` files generated by Table 1.

After aggregating Table 1, verify the generalized metrics with:

```bash
python tools/compare_expected.py \
  --results outputs/voc/summary.json \
  --table table4_generalized_voc
```

---

## 12. Reproducing Table 5 — Component Ablation

Run:

```bash
NUM_GPUS=4 bash scripts/reproduce_table5_ablation.sh
```

This evaluates VOC Split 1, seed 0, at 1/3/5 shots.

Component mapping:

| Row | SAIA | DC | SPDI |
| --- | :---: | :---: | :---: |
| Baseline | ✗ | ✗ | ✗ |
| +DC | ✗ | ✓ | ✗ |
| +SAIA | ✓ | ✗ | ✗ |
| +DC+SAIA | ✓ | ✓ | ✗ |
| +SPDI | ✗ | ✗ | ✓ |
| Full EADA | ✓ | ✓ | ✓ |

The +SPDI row reuses the Baseline checkpoint and only enables frozen inference-time semantic integration. Full EADA reuses the DC+SAIA checkpoint and adds SPDI at evaluation. Thus semantic integration is not allowed to change detector training.

Expected novel AP50:

| Row | 1-shot | 3-shot | 5-shot |
| --- | ---: | ---: | ---: |
| Baseline | 53.24 | 60.69 | 66.75 |
| +DC | 54.81 | 60.32 | 68.31 |
| +SAIA | 53.55 | 60.67 | 66.39 |
| +DC+SAIA | 54.79 | 61.58 | 67.18 |
| +SPDI | 66.54 | 70.04 | 73.01 |
| Full EADA | 67.86 | 71.05 | 73.73 |

---

## 13. Reproducing Table 6 — SPDI Placement and Fusion

```bash
bash scripts/reproduce_table6_spdi.sh
```

The placement study uses one frozen 1-shot DC+SAIA checkpoint:

```text
Detector-only
Post-NMS SPDI
Pre-NMS SPDI
```

The fusion study uses one frozen 10-shot DC+SAIA checkpoint:

```text
Detector-only
Normalized pre-NMS fusion
Absolute-score pre-NMS fusion  <- final EADA
```

Controls are implemented by:

```text
TEST.SPDI_PLACEMENT = pre_nms | post_nms
TEST.SPDI_FUSION_MODE = absolute | normalized
```

Expected values are stored under `table6_spdi_design` in `expected/paper_results.json`.

---

## 14. Reproducing Table 7 — Direct Mechanism Diagnostics

```bash
bash scripts/reproduce_table7_diagnostics.sh
```

This directly measures the mechanisms instead of only final AP.

### SAIA proposal coverage

`tools/saia_proposal_recall.py` reports:

- Recall@100 at IoU >= 0.50;
- Recall@100 at IoU >= 0.75;
- mean best proposal IoU for novel ground-truth objects.

Paper targets:

```text
                         Control    SAIA
Recall@100, IoU >= .50    88.41     88.62
Recall@100, IoU >= .75    56.86     57.33
Mean best IoU             70.84     70.97
```

### DC gradient diagnostic

`tools/dc_gradient_diagnostic.py` applies standard CE and DC to the **same** sampled logits and measures background-RoI classification gradients.

Paper targets (x 1e-3):

```text
                         CE       DC
Absent-class gradient   1.678    0.000
Supported gradient      7.708    7.876
```

---

## 15. Reproducing Table 8 — Sensitivity

```bash
NUM_GPUS=4 bash scripts/reproduce_table8_sensitivity.sh
```

The script reproduces:

- SAIA alpha = 0.1 / 0.2 / 0.3 / 0.4 / 0.5;
- SPDI lambda_N = 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0;
- SPDI temperature = 0.005 / 0.01 / 0.02 / 0.05;
- SPDI top-k = 25 / 50 / 100 / 200.

The SPDI sweeps reuse the same frozen detector checkpoint; only inference configuration changes.

---

## 16. Reproducing Table 9 — Efficiency

Run on an RTX 4070 Ti to match the paper hardware:

```bash
bash scripts/reproduce_table9_efficiency.sh
```

The profiler records:

- detector parameter count;
- CLIP parameter count;
- detector GFLOPs;
- CLIP image-encoder GFLOPs per proposal crop;
- top-k upper-bound GFLOPs;
- peak inference memory;
- mean latency;
- FPS;
- GPU model and PyTorch version.

Paper reference:

| Setting | Det. param. M | CLIP param. M | GFLOPs | Infer mem GiB | Latency ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 64.7 | 0.0 | 192.1 | 2.34 | 63.3 | 15.81 |
| DC+SAIA | 68.0 | 0.0 | 200.1 | 2.36 | 68.1 | 14.68 |
| Full EADA | 68.0 | 149.6 | 1886.9 | 3.22 | 806.3 | 1.24 |

Latency and memory are hardware-sensitive; compare these values only on matched hardware/software.

---

## 17. Output Files

A normal VOC cell produces:

```text
outputs/voc/split1/1shot/seed0/
├── run_protocol.json
├── train/
│   ├── config.yaml
│   ├── metrics.json
│   └── model_final.pth
└── eval/
    ├── config.yaml
    ├── evaluation_results.json
    └── inference/
```

COCO uses the analogous directory:

```text
outputs/coco/10shot/seed2/
```

The output tree deliberately separates training and SPDI evaluation so that inference-time semantic integration cannot silently alter the detector checkpoint.

---

## 18. Reproducibility and Random Seeds

The main paper uses seeds:

```text
0, 1, 2
```

The dataset support split **and** Detectron2/PyTorch random seed must use the same seed ID. The release scripts override both together.

Do not change only `SEED` while retaining a different `seedX` dataset name.

VOC support loading preserves the benchmark sampling behavior when a support image contains more than K valid instances. This means deterministic reproduction requires both the same support text files and the same random seed.

---

## 19. Reviewer Validation Modes

### Level A — source/package validation

No dataset or GPU training:

```bash
bash scripts/verify_release.sh
```

This checks unit tests, source syntax, config invariants, release integrity, method naming, and shell syntax.

### Level B — environment + dataset validation

```bash
python tools/check_environment.py
python tools/check_data.py
```

### Level C — one-cell scientific reproduction

```bash
NUM_GPUS=4 bash scripts/run_voc_cell.sh 1 1 0
```

This is the recommended reviewer smoke reproduction because it exercises all three EADA mechanisms.

### Level D — complete paper reproduction

```bash
NUM_GPUS=4 bash scripts/reproduce_table1_voc.sh
NUM_GPUS=4 bash scripts/reproduce_table2_coco.sh
NUM_GPUS=4 bash scripts/reproduce_table3_fsis.sh
NUM_GPUS=4 bash scripts/reproduce_table5_ablation.sh
bash scripts/reproduce_table6_spdi.sh
bash scripts/reproduce_table7_diagnostics.sh
NUM_GPUS=4 bash scripts/reproduce_table8_sensitivity.sh
bash scripts/reproduce_table9_efficiency.sh
```

Full reproduction is computationally expensive because Tables 1 and 2 alone contain 63 training/evaluation cells across three seeds.

---

## 20. Release Self-Audit

Run:

```bash
python tools/audit_release.py
```

The audit checks:

- required implementation/reproduction files;
- Python syntax;
- YAML structure and schedules;
- absence of absolute cluster paths;
- absence of symlinks inside the release;
- absence of generated checkpoints/logs;
- absence of retired internal method identifiers;
- file inventory and release structure.

The public source tree intentionally uses only the paper terminology **EADA / SAIA / DC / SPDI**.

---

## 21. Important Fidelity Notes

### 21.1 Active C4 width

The actual Detectron2 ResNet-101-C4 pre-RPN `res4` feature used by the experiment checkpoints has **1024 channels**. The released SAIA implementation therefore uses:

```text
1024 -> 256 -> 1024
```

No artificial 1024→2048 bridge is inserted.

### 21.2 Two-stream batch semantics

`SOLVER.IMS_PER_BATCH=4` is the batch size of **each** few-shot training stream. One optimizer step concatenates:

```text
4 support images + 4 base images = 8 images
```

This is preserved exactly by `_EADATwoStreamLoader`.

### 21.3 No training-time CLIP path

SPDI is strictly inference-only. The final detector training objective is the detector/RPN losses plus DC; no CLIP loss is optimized.

### 21.4 Base/background preservation

SPDI modifies only novel-category score coordinates. Base-class scores, background score, proposal coordinates, and mask logits are unchanged.

---

## 22. Troubleshooting

### `No module named mmcv._ext` or deformable attention import failure

Your MMCV installation lacks compiled CUDA/C++ ops. Build MMCV 1.7.2 with `MMCV_WITH_OPS=1`. See [INSTALL.md](INSTALL.md).

### Detectron2 build/import mismatch

The verified stack uses the legacy Detectron2 0.3 API with PyTorch 2.2.2/CUDA 12.1. Rebuild Detectron2 after installing the final PyTorch version.

### COCO support JSON cannot be found

Run:

```bash
bash scripts/setup_splits.sh
```

and confirm:

```bash
echo $DETECTRON2_DATASETS
python tools/check_data.py
```

### COCO `trainvalno5k.json` / `5k.json` missing

Run:

```bash
bash scripts/download_coco_datasplit.sh
```

or copy the standard TFA/FsDet `cocosplit/datasplit` files manually.

### SPDI cannot download CLIP weights

Use a local file:

```bash
export EADA_CLIP_CHECKPOINT=/absolute/path/to/ViT-B-16.pt
```

### CUDA out of memory

Do not silently change `SOLVER.IMS_PER_BATCH` if comparing with paper numbers. Instead reproduce on four GPUs as documented. For code-path debugging only, smaller batches are acceptable but should not be treated as paper reproduction.

---

## 23. What Is and Is Not Included

Included:

- complete EADA source;
- VOC/COCO registration and evaluators;
- exact paper seed-0/1/2 few-shot support files;
- base/few-shot/FSIS configs;
- checkpoint surgery;
- full benchmark launchers;
- component and mechanism diagnostics;
- sensitivity and efficiency scripts;
- expected paper values;
- unit tests and integrity audit.

Not included:

- raw PASCAL VOC images/annotations;
- raw MS COCO images;
- the ~500 MB COCO `trainvalno5k.json` benchmark annotation;
- ImageNet pretrained weights;
- CLIP weights;
- trained detector checkpoints;
- private cluster logs.

These are excluded because of dataset/model licensing and archive size, not because they are required proprietary resources.

---

## 24. Attribution and License

See:

- [LICENSE](LICENSE)
- [ATTRIBUTION.md](ATTRIBUTION.md)
- [PROVENANCE.md](PROVENANCE.md)

The release builds on the Detectron2/DeFRCN-style transfer-based FSOD codebase and uses MMCV/MMDetection for deformable attention and OpenCLIP for frozen CLIP ViT-B/16 inference. Dependencies are installed separately and retain their own licenses.

---

## 25. Final Reproduction Checklist

Before comparing a result with the paper, verify all of the following:

- environment check passes;
- exact seed-0/1/2 support files are installed;
- correct VOC/COCO dataset layout is used;
- R101 initialization file is present;
- base detector has been trained and head-expanded;
- training uses 4 GPUs for the paper protocol;
- support/base two-stream training is enabled;
- VOC uses SAIA α=0.2 and RoI gradient scale 0.001;
- COCO uses SAIA α=0.3 and RoI gradient scale 0.01;
- DC is enabled only during training;
- SPDI is disabled during training and enabled during evaluation;
- SPDI uses top-100, τ=0.01, λ_N=0.7, absolute pre-NMS fusion;
- the support dataset seed matches the model random seed;
- the main-table run is selected by novel AP50 from seeds 0/1/2;
- all reported metrics in one row come from the same selected run.

For a more command-oriented walkthrough, see [REPRODUCE.md](REPRODUCE.md).
