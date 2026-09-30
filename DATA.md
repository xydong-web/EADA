# Dataset preparation

EADA uses the standard transfer-based few-shot detection protocols for PASCAL
VOC and MS COCO. Raw datasets are not redistributed. The release includes the
exact few-shot support split files for seeds 0, 1, and 2, which are the three
seeds used for the paper tables.

## Expected root

Set one dataset root before running any command:

```bash
export DETECTRON2_DATASETS=/absolute/path/to/eada_datasets
```

The expected layout is:

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
├── vocsplit/
│   ├── seed0/
│   ├── seed1/
│   └── seed2/
├── coco/
│   ├── trainval2014/
│   └── val2014/
└── cocosplit/
    ├── datasplit/
    │   ├── trainvalno5k.json
    │   └── 5k.json
    ├── seed0/
    ├── seed1/
    └── seed2/
```

## Raw PASCAL VOC

Prepare VOC 2007 and VOC 2012 in the standard VOC layout. EADA uses the 2007
and 2012 `trainval` sets for training and the VOC 2007 `test` set for
evaluation.

## Raw MS COCO

The experiment code uses the COCO 2014 train/val layout used by the TFA FSOD
benchmark:

```text
coco/trainval2014/
coco/val2014/
```

If your standard COCO download contains only `train2014/` and `val2014/`, build
the combined image view required by the benchmark without duplicating images:

```bash
bash scripts/prepare_coco_images.sh
```

The script creates `coco/trainval2014/` as a directory of symbolic links to the
COCO 2014 train and validation images. COCO image IDs are disjoint across those
two source directories, so no filename rewriting is required.

The standard benchmark split requires `trainvalno5k.json` and `5k.json`.
These files are not bundled because `trainvalno5k.json` is over 500 MB. They
are the standard TFA/FsDet split files.

The helper below downloads the benchmark files from the public FsDet/TFA split
host when network access is available:

```bash
bash scripts/download_coco_datasplit.sh
```

If the host is unavailable, obtain the same `cocosplit/datasplit` files from
the official TFA/FsDet few-shot dataset package and place them at the paths
shown above.

## Bundled support splits

The exact seed 0/1/2 support files used by the paper are stored in:

```text
splits/voc/seed{0,1,2}/
splits/coco/seed{0,1,2}/
```

Install them into the dataset root with:

```bash
bash scripts/setup_splits.sh
```

The script copies the support files by default so the prepared dataset does not
depend on the location of the extracted source archive. Pass `COPY_SPLITS=0`
to use symbolic links instead.

## Validate data before training

Run:

```bash
python tools/check_data.py
```

The checker verifies directory structure, the presence of the two COCO
benchmark JSON files, and the expected counts of bundled few-shot split files
for all three seeds.
For a VOC-only or COCO-only reviewer run, use `--dataset voc` or
`--dataset coco` so unrelated raw data are not required.

