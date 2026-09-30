# Provenance

This submission package was distilled from the current research working tree
and rewritten so that all public method identifiers follow the terminology of
the EADA manuscript: EADA, SAIA, DC, and SPDI.

The detector data loaders and evaluators derive from the DeFRCN-style
Detectron2 codebase summarized in SOURCE_MANIFEST.json. The EADA-specific
implementation was traced against the final training and evaluation paths
rather than reconstructed only from manuscript prose.

Important fidelity choices:

- SAIA uses the native R101-C4 res4 width produced by the experiment code
  (1024 channels), a 1024 -> 256 -> 1024 bottleneck, and the same complete-C4
  validity convention used by the completed experiment path.
- DC uses multiplicative image-conditioned background gating without removing
  inactive logits from the softmax vector.
- SPDI uses direct novel-coordinate score fusion before NMS and frozen OpenAI
  CLIP ViT-B/16 weights.
- The released detector head is protocol-aware. Base pretraining learns a
  residual mapping with a trainable classifier; VOC few-shot adaptation keeps
  that mapping and freezes the expanded classifier; COCO few-shot adaptation
  uses the standard trainable linear classifier. These are baseline/checkpoint
  compatibility choices rather than additional EADA mechanisms.
- The few-shot training loader reproduces the support-plus-base two-stream
  optimization path.
- The optional FSIS path freezes the EADA detector and optimizes only a
  class-agnostic mask head.

The reviewer package includes the exact seed-0/1/2 few-shot support annotation
files used by the reported experiments. Raw VOC/COCO images, the large COCO
base/test annotation files, model weights, experiment outputs, and
machine-specific paths are not included.
