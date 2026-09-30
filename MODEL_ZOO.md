# Model initialization and checkpoints

Model weights are intentionally not included in the source archive. The paper
checkpoints are large (hundreds of MB to several GB when optimizer state is
present), while all models can be regenerated from the documented training
commands.

For reviewer convenience, the submission also prepares a **separate
initialization bundle**. It contains only the base-expanded/model-initialization
states required to launch the paper adaptation runs; it does not contain final
few-shot EADA checkpoints.

The companion archive is named `EADA_Reviewer_Initialization_Weights.zip` and
uses the same top-level `EADA/` directory as the source archive. It can therefore
be extracted directly on top of `EADA_Reviewer_Reproduction_Code.zip`.

## Exact reviewer initialization bundle

Extract the companion bundle into the repository root. The files and exact
validated file sizes are:

| Target path | Bytes |
| --- | ---: |
| `weights/voc_split1_all.pth` | 259869626 |
| `weights/voc_split2_all.pth` | 259869626 |
| `weights/voc_split3_all.pth` | 259869626 |
| `weights/coco_all.pth` | 262328442 |
| `weights/coco_base_mask_init.pth` | 280699296 |

The first four files are the exact expanded base detector states used to start
few-shot adaptation. `coco_base_mask_init.pth` is the exact class-agnostic base
mask initialization used by the FSIS extension.

Using these files is the recommended reviewer path. Rebuilding the base models
from ImageNet remains supported below, but the historical classifier-surgery
step used random novel-row initialization. Therefore a newly generated
expanded checkpoint is scientifically equivalent but is not guaranteed to be
bitwise identical to the archived paper initialization.

After extracting the bundle, check that every file exists with the expected
size:

```bash
python tools/check_weights.py
```

## ImageNet R101 initialization

Base training starts from the Detectron-style ImageNet-pretrained ResNet-101
checkpoint `R-101.pkl`. The file used for the validated runs has:

```text
size:   178431595 bytes
```

Place it at:

```text
weights/R-101.pkl
```

The corresponding torchvision ResNet-101 checkpoint is used only for auxiliary
checks.

## Frozen CLIP used by SPDI

SPDI uses OpenAI CLIP ViT-B/16 through OpenCLIP. If
`TEST.SPDI_CHECKPOINT` is empty, OpenCLIP downloads/loads its `openai`
pretrained weight automatically. For completely offline reproduction, point
the config to a local checkpoint:

```bash
export EADA_CLIP_CHECKPOINT=/absolute/path/to/ViT-B-16.pt
```

The local file used for the validated paper runs has:

```text
size:   approximately 335 MiB
```

## Base detector checkpoints

The reproduction scripts create:

```text
outputs/base/voc_split1/model_final.pth
outputs/base/voc_split2/model_final.pth
outputs/base/voc_split3/model_final.pth
outputs/base/coco/model_final.pth
```

They then expand the base classifier/regressor to the joint base+novel label
space using `tools/expand_checkpoint.py`.

Important protocol detail: base pretraining uses the `residual_trainable`
head so that the residual mapping is learned from base categories. VOC
few-shot adaptation switches to `residual_frozen`: the mapping remains in
the forward path while the expanded classifier is frozen. COCO few-shot
adaptation uses `standard`, i.e. the ordinary trainable Fast R-CNN
classifier without the residual mapping in the forward path. These settings
are encoded in the released configs and should not be changed when reproducing
the reported numbers.

