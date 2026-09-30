# Installation

This document records the environment used to validate the EADA release. The
commands below are intended for Linux systems with NVIDIA GPUs.

## Verified software stack

The final experiment environment used:

| Package | Version |
| --- | --- |
| Python | 3.9.23 |
| PyTorch | 2.2.2 + CUDA 12.1 |
| torchvision | 0.17.2 + CUDA 12.1 |
| Detectron2 | 0.3 legacy API |
| MMCV | 1.7.2 |
| MMDetection | 2.28.2 |
| OpenCLIP | 2.24.0 |
| NumPy | 1.26.4 |

MMDetection 2.28.2 requires the MMCV 1.x family; do not replace it with
MMCV 2.x. Detectron2 is also intentionally pinned to the legacy 0.3 API used
by the experiments.

## Recommended installation

Create the base environment:

```bash
conda env create -f environment.yml
conda activate eada
```

Install MMCV 1.7.2 with CUDA operators. The deformable-attention operator is
required by SAIA, so an `mmcv-lite` installation is insufficient. A source
build is the most reliable option for PyTorch 2.2/CUDA 12.1:

```bash
git clone --branch v1.7.2 --depth 1 https://github.com/open-mmlab/mmcv.git third_party/mmcv
cd third_party/mmcv
MMCV_WITH_OPS=1 pip install -v -e .
cd ../..
```

Install MMDetection only after MMCV is present. The experiment stack uses
MMDetection 2.28.2 with the MMCV 1.x API family:

```bash
python -m pip install --no-deps mmdet==2.28.2
```

Install the legacy Detectron2 API used by this release:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.3'
```

If the Detectron2 build fails because of a CUDA/compiler mismatch, make sure
that the CUDA toolkit visible to `nvcc` is compatible with the CUDA version of
the installed PyTorch build. The release was validated with a CUDA 12.1
PyTorch build.

Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` deliberately does not install MMCV or Detectron2. Those two
packages contain compiled extensions and must remain bound to the PyTorch/CUDA
toolchain selected above.

## Environment self-check

From the repository root:

```bash
python tools/check_environment.py
```

The command checks versions, imports the deformable-attention CUDA operator,
constructs the EADA config, and reports whether CUDA is available.

## Notes on supported hardware

The paper training runs used four GPUs for base training and few-shot
adaptation. The global support-stream batch is 4 in few-shot adaptation; the
two-stream loader adds a second base-data batch of the same size, so each
optimizer step contains 8 images in total. A one-GPU run is supported for
debugging, but the paper numbers should be reproduced with the documented
four-GPU protocol.

SPDI evaluation is performed on one GPU because the frozen CLIP proposal crop
stage is independent of detector training.

