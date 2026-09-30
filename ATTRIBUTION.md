# Attribution

- The supplied repository carries the MIT license in LICENSE, copyright
  WuShuang1998 (2022). Its DeFRCN-derived dataset metadata, few-shot loaders and
  VOC/COCO evaluators are adapted under that notice. Detector decoupling,
  dual classification/localization streams, DC, SAIA wiring and score fusion
  derive from the current source tree, as summarized in SOURCE_MANIFEST.json.
- DeFRCN: https://github.com/er-muyue/DeFRCN (ICCV 2021).
- Detectron2: https://github.com/facebookresearch/detectron2 (Apache-2.0).
  Installed separately; supplies the ResNet, RPN, RoI blocks and training loop.
- MMCV / MMDetection: https://github.com/open-mmlab/mmcv and
  https://github.com/open-mmlab/mmdetection (Apache-2.0). Installed separately;
  supply the four-layer deformable-attention encoder implementation.
- OpenCLIP: https://github.com/mlfoundations/open_clip (MIT). Installed
  separately and used to load the OpenAI-pretrained CLIP ViT-B/16 weights,
  preprocessing, and tokenizer used by SPDI.

Dependencies, pretrained weights, raw VOC/COCO images, and full benchmark
annotations are not redistributed. The reviewer package does include the
seed-0/1/2 few-shot support annotation files required to identify the exact
paper subsets; those files follow the established TFA/FsDet benchmark layout
and remain subject to the applicable upstream dataset/benchmark terms. Keep
LICENSE and this attribution with redistributed source. See PROVENANCE.md for
implementation adaptations.
