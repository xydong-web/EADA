# Submission verification

Validation date: 2026-09-28.

The experiment validation records below were supplied with the original
reviewer package. They do not represent a new benchmark run for GitHub
publication.

The release was validated in the existing experiment environment:

- Python 3.9.23
- PyTorch 2.2.2 + CUDA 12.1 build
- torchvision 0.17.2
- NumPy 1.26.4
- Detectron2 legacy API used by the project
- MMCV 1.7.2
- MMDetection 2.28.2
- open_clip_torch 2.24.0

No benchmark training jobs were submitted during packaging.

## Static and unit validation

From the release root:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py
bash -n scripts/train.sh scripts/eval.sh scripts/train_fsis.sh
```

Result: all 14 release tests pass. They cover:

1. DC image-conditioned gating and zero absent-class gradient.
2. DC empty/invalid-input behavior.
3. Direct SPDI novel-score fusion with base/background preservation.
4. SPDI top-100 proposal selection and exact paper prompt templates.
5. Frozen CLIP crop-path behavior through an offline test double.
6. Native-width SAIA encoder wiring and the complete-C4 reference geometry
   used by the experiment checkpoints.
7. Base-to-joint classifier checkpoint expansion.
8. FSIS detector/mask checkpoint composition.
9. EADA config schedules, two-stream settings, and FSIS architecture.
10. SAIA-before-RPN and SPDI-before-NMS execution order.
11. Retired-identifier and release-artifact negative controls.
12. README command hygiene, including rejection of patch-artifact command text.
13. Normalized-SPDI control behavior, including preservation of total novel
    detector probability mass.
14. Exact seed-0/1/2 VOC/COCO support-split inventory.

The release audit checks required files, support-split counts, Python syntax,
and YAML configuration. It rejects machine-specific paths, symlinks,
generated artifacts, and retired internal method identifiers.

The dataset preflight was also run against the complete local benchmark tree.
It verified all three bundled support seeds, the VOC/COCO directory structure,
and the required COCO benchmark-annotation files.

The companion reviewer initialization bundle was checked with
`tools/check_weights.py`. All five initialization checkpoints match the sizes
in MODEL_ZOO.md. The published checker validates file presence and sizes only;
it does not establish bitwise identity. The bundle provides the expanded
base/model initialization states without shipping final EADA few-shot
checkpoints.

Reviewer-facing table verification utilities are included as well. The main
VOC/COCO runners aggregate the reproduced seeds and compare them against the
stored paper targets, while Tables 3 and 5--9 use
`tools/verify_auxiliary.py`. Table 9 is non-strict by default because latency
and memory are hardware-dependent.

## Detector checkpoint-compatibility validation

The submission EADA detector was constructed on CPU and compared with final
VOC Split 1 one-shot and COCO 10-shot experiment checkpoints.

```text
source checkpoint tensors: 615
release model tensors:      613
common tensor names:        611
shape mismatches:             0
source-only tensors:          4
release-only tensors:         2
```

The four source-only tensors are unused feature-memory queue buffers. The two
release-only tensors are Detectron2 pixel mean/std buffers. All detector,
SAIA, affine-decoupling, residual classifier-mapping, classifier, and box
regression tensor names and shapes match the experiment checkpoint.

The constructed release model also verifies:

```text
SAIA input projection:  (256, 1024, 1, 1)
SAIA output projection: (1024, 256, 1, 1)
linear classifier trainable: false
```

The COCO few-shot protocol was checked separately because its paper results
use the standard trainable Fast R-CNN classifier rather than the VOC
residual/frozen head. Base pretraining for both benchmarks uses the
residual/trainable head so the mapping is learned before the benchmark-specific
few-shot protocol is selected. The corresponding COCO compatibility result is:

```text
source checkpoint tensors: 614
release model tensors:      612
common tensor names:        610
shape mismatches:             0
source-only tensors:          4
release-only tensors:         2
residual mapping present:  false / false (release / source)
linear classifier trainable: true
```

Thus the release selects the detector-head protocol explicitly through
`MODEL.EADA.ROI_PREDICTOR_MODE`: base-training configs use
`residual_trainable`, VOC few-shot configs use `residual_frozen`, and
COCO few-shot configs use `standard`.

The base-stage release models were also compared with the actual base
checkpoints used to initialize the paper experiments:

```text
VOC base:
  release tensors:       542
  source tensors:        540
  common tensors:        540
  shape mismatches:        0
  residual mapping:     present / present
  classifier trainable: true

COCO base:
  release tensors:       542
  source tensors:        540
  common tensors:        540
  shape mismatches:        0
  residual mapping:     present / present
  classifier trainable: true
```

The only release-only tensors in both base comparisons are Detectron2 pixel
mean/std buffers.

## FSIS construction validation

The COCO 10-shot FSIS config was constructed on CPU. The resulting model has a
class-agnostic Mask R-CNN head with four 256-channel convolution layers and a
14 x 14 RoIAlign pooler. In mask-only mode, 12 parameter tensors are trainable
and every trainable tensor belongs to roi_heads.mask_head; no detector
parameter remains trainable.

CPU construction smoke tests also passed for VOC base, VOC few-shot, COCO base,
COCO few-shot, and COCO FSIS configs. They verify the intended predictor modes:

```text
VOC base:      residual_trainable, mapping present, classifier trainable
VOC few-shot:  residual_frozen,    mapping present, classifier frozen
COCO base:     residual_trainable, mapping present, classifier trainable
COCO few-shot: standard,           mapping absent,  classifier trainable
COCO FSIS:     standard, detector frozen, only mask-head tensors trainable
```

## Two-stream protocol validation

The release two-stream loader was exercised with synthetic loaders configured
to emit four support images and four base images. One EADA optimization batch
contains eight images total:

```text
support images: 4
base images:    4
combined model-forward batch: 8
```

This matches the final experiment driver behavior and is why README.md flags
the manuscript wording that currently calls the global batch size four.

## Scope and limitations

The validation above checks source fidelity, construction, configuration,
state-dictionary compatibility, component invariants, and packaging integrity.
It does not rerun the full VOC/COCO benchmark matrix, does not download CLIP
weights, and does not execute the deformable-attention CUDA kernel on a GPU.
Benchmark metrics remain those produced by the already completed experiment
runs in the research workspace.

