import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from eada.losses import dc_logits, dc_loss
from eada.saia import apply_saia
from eada.spdi import PROMPTS, SPDI, fuse_novel, select_proposals
from tools.audit_release import FORBIDDEN, audit, load_config, violations
from tools.expand_checkpoint import expand_state
from tools.merge_fsis_checkpoint import merge_states


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_dc_per_image_and_foreground(self):
        logits = torch.arange(20, dtype=torch.float32).reshape(5, 4).requires_grad_()
        labels = torch.tensor([0, 3, 3, 2, 3])
        gated = dc_logits(logits, labels, [2, 3], [[0], [2]])
        torch.testing.assert_close(gated[[0, 3]], logits[[0, 3]])
        self.assertEqual(gated[1].tolist(), [4, 0, 0, 7])
        self.assertEqual(gated[2].tolist(), [0, 0, 10, 11])
        dc_loss(logits, labels, [2, 3], [[0], [2]]).backward()
        self.assertEqual(logits.grad[1, 1].item(), 0)
        self.assertNotEqual(logits.grad[0, 1].item(), 0)

    def test_dc_validation_empty(self):
        with self.assertRaises(ValueError):
            dc_logits(torch.ones(2, 3), torch.tensor([2, 2]), [1], [[0]])
        with self.assertRaises(ValueError):
            dc_logits(torch.ones(2, 3), torch.tensor([2, 2]), [2], [[4]])
        loss = dc_loss(
            torch.empty(0, 3),
            torch.empty(0, dtype=torch.long),
            [0],
            [[]],
        )
        self.assertEqual(loss.item(), 0)

    def test_spdi_direct_novel_fusion(self):
        detector = torch.tensor(
            [[0.2, 0.1, 0.3, 0.4], [0.1, 0.2, 0.5, 0.2]]
        )
        semantic = torch.tensor(
            [[0.2, 0.7, 0.1], [0.3, 0.1, 0.6]]
        )
        fused = fuse_novel(detector, semantic, [1, 2], 0.7)
        torch.testing.assert_close(
            fused[:, [0, 3]],
            detector[:, [0, 3]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fused[:, 1:3],
            0.7 * detector[:, 1:3] + 0.3 * semantic[:, 1:3],
        )
        torch.testing.assert_close(
            fuse_novel(detector, semantic, [1, 2], 1.0),
            detector,
        )
        with self.assertRaises(ValueError):
            fuse_novel(detector, semantic, [1], 1.1)

    def test_spdi_selection_and_prompts(self):
        self.assertEqual(
            fuse_novel(torch.empty(0, 4), torch.empty(0, 3), [1]).shape,
            (0, 4),
        )
        scores = torch.arange(150).float()[:, None].repeat(1, 4)
        self.assertEqual(
            set(select_proposals(scores).tolist()),
            set(range(50, 150)),
        )
        self.assertEqual(select_proposals(scores[:0]).numel(), 0)
        self.assertEqual(PROMPTS, ("a photo of a {}", "a clean photo of a {}"))

    def test_spdi_normalized_control_preserves_novel_mass(self):
        detector = torch.tensor([[0.2, 0.1, 0.3, 0.4]])
        semantic = torch.tensor([[0.7, 0.1, 0.2]])
        fused = fuse_novel(
            detector,
            semantic,
            [1, 2],
            0.7,
            mode="normalized",
        )
        torch.testing.assert_close(
            fused[:, [0, 3]], detector[:, [0, 3]], rtol=0, atol=0
        )
        torch.testing.assert_close(
            fused[:, [1, 2]].sum(1),
            detector[:, [1, 2]].sum(1),
        )
        with self.assertRaises(ValueError):
            fuse_novel(detector, semantic, [1, 2], mode="invalid")

    def test_saia_operates_at_native_c4_width(self):
        captured = {}

        def encoder(**kwargs):
            captured.update(kwargs)
            return kwargs["query"]

        model = SimpleNamespace(
            lateral_conv_in=torch.nn.Conv2d(1024, 256, 1),
            lateral_conv_out=torch.nn.Conv2d(256, 1024, 1),
            encoder=encoder,
            level_embeds=torch.zeros(1, 256),
            positional_encoding=lambda mask: torch.zeros(
                mask.shape[0], 256, *mask.shape[-2:]
            ),
            saia_alpha=0.2,
        )
        images = SimpleNamespace(
            tensor=torch.zeros(2, 3, 64, 64),
            image_sizes=[(64, 64), (32, 48)],
        )
        features = torch.randn(2, 1024, 4, 4, requires_grad=True)
        output = apply_saia(model, features, images)
        self.assertEqual(output.shape, features.shape)
        self.assertEqual(captured["query"].shape, (16, 2, 256))
        self.assertEqual(captured["reference_points"].shape, (2, 16, 1, 2))
        self.assertEqual(
            captured["query_key_padding_mask"][1].sum().item(),
            0,
        )
        torch.testing.assert_close(
            captured["valid_ratios"][1, 0],
            torch.tensor([1.0, 1.0]),
        )

    def test_checkpoint_expansion(self):
        state = {}
        for head, rows in (("cls_score", 3), ("bbox_pred", 8)):
            for suffix in ("weight", "bias"):
                key = "roi_heads.box_predictor." + head + "." + suffix
                state[key] = (
                    torch.arange(rows * 2).float().reshape(rows, 2)
                    if suffix == "weight"
                    else torch.arange(rows).float()
                )
        result = expand_state(
            state,
            ["base_a", "base_b"],
            ["novel", "base_b", "base_a"],
        )
        key = "roi_heads.box_predictor.cls_score.weight"
        torch.testing.assert_close(result[key][2], state[key][0])
        torch.testing.assert_close(result[key][-1], state[key][-1])
        key = "roi_heads.box_predictor.bbox_pred.weight"
        torch.testing.assert_close(result[key][8:12], state[key][:4])

    def test_fsis_checkpoint_composition(self):
        detector = {
            "backbone.x": torch.tensor([1.0]),
            "roi_heads.mask_head.old": torch.tensor([0.0]),
        }
        mask = {
            "roi_heads.mask_head.conv": torch.tensor([2.0]),
            "backbone.x": torch.tensor([9.0]),
        }
        merged, keys = merge_states(detector, mask)
        self.assertEqual(keys, ["roi_heads.mask_head.conv"])
        torch.testing.assert_close(merged["backbone.x"], detector["backbone.x"])
        torch.testing.assert_close(
            merged["roi_heads.mask_head.conv"],
            mask["roi_heads.mask_head.conv"],
        )
        self.assertNotIn("roi_heads.mask_head.old", merged)

    def test_spdi_mock_open_clip_crops_and_unselected(self):
        class FakeClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.crop_batches = []

            def encode_text(self, tokens):
                return torch.eye(3).repeat_interleave(2, dim=0)

            def encode_image(self, crops):
                self.crop_batches.append(crops.shape)
                return torch.tensor([0.1, 0.2, 0.3]).repeat(len(crops), 1)

        clip_model = FakeClip()
        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                clip_model,
                None,
                lambda image: torch.zeros(3, 224, 224),
            ),
            tokenize=lambda text: torch.zeros(len(text), 8, dtype=torch.long),
        )
        with patch.dict(sys.modules, {"open_clip": fake_open_clip}):
            calibrator = SPDI(
                ["base", "novel_a", "novel_b"],
                [1, 2],
                "cpu",
            )
        self.assertFalse(clip_model.training)
        self.assertFalse(clip_model.anchor.requires_grad)
        detector = torch.tensor([0.1, 0.2, 0.3, 0.4]).repeat(120, 1)
        detector[:, 0] = torch.arange(120).float() / 1000
        selected = select_proposals(detector)
        boxes = torch.tensor([0, 0, 8, 8]).float().repeat(120, 1)
        boxes[selected[0]] = 0
        result = calibrator(torch.zeros(3, 16, 16), boxes, detector)
        untouched = sorted(set(range(120)) - set(selected.tolist()))
        torch.testing.assert_close(
            result[untouched],
            detector[untouched],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result[selected[0]],
            detector[selected[0]],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result[:, [0, 3]],
            detector[:, [0, 3]],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            sum(shape[0] for shape in clip_model.crop_batches),
            99,
        )
        self.assertTrue(
            all(shape[1:] == (3, 224, 224) for shape in clip_model.crop_batches)
        )

    def test_audit_negative_controls(self):
        for token in FORBIDDEN:
            self.assertTrue(violations("eada/config.py", "WITH" + token))
            self.assertTrue(
                violations("configs/" + token.lower() + ".yaml", "")
            )

    def test_config_grid_schedules_and_audit(self):
        audit()
        self.assertEqual(len(list((ROOT / "configs").rglob("*.yaml"))), 33)
        voc = load_config(ROOT / "configs/voc/split1_10shot.yaml")
        self.assertEqual(voc["MODEL"]["ROI_HEADS"]["NUM_CLASSES"], 20)
        self.assertEqual(voc["SOLVER"]["IMS_PER_BATCH"], 4)
        self.assertEqual(voc["SOLVER"]["BASE_LR"], 0.0025)
        self.assertEqual(voc["SOLVER"]["STEPS"], [2400])
        self.assertEqual(voc["SOLVER"]["MAX_ITER"], 3000)
        self.assertTrue(voc["DATASETS"]["EADA_TWO_STREAM"])
        self.assertEqual(
            voc["MODEL"]["EADA"]["ROI_PREDICTOR_MODE"],
            "residual_frozen",
        )
        voc_base = load_config(ROOT / "configs/voc/base1.yaml")
        self.assertEqual(
            voc_base["MODEL"]["EADA"]["ROI_PREDICTOR_MODE"],
            "residual_trainable",
        )
        self.assertEqual(voc_base["SOLVER"]["IMS_PER_BATCH"], 32)
        self.assertEqual(voc_base["SOLVER"]["STEPS"], [10000, 13300])
        self.assertEqual(voc_base["SOLVER"]["MAX_ITER"], 15000)
        coco = load_config(ROOT / "configs/coco/30shot.yaml")
        self.assertEqual(coco["SOLVER"]["STEPS"], [8000])
        self.assertEqual(coco["SOLVER"]["MAX_ITER"], 9600)
        self.assertEqual(coco["MODEL"]["EADA"]["SAIA_ALPHA"], 0.3)
        self.assertEqual(
            coco["MODEL"]["EADA"]["ROI_PREDICTOR_MODE"],
            "standard",
        )
        coco_base = load_config(ROOT / "configs/coco/base.yaml")
        self.assertEqual(
            coco_base["MODEL"]["EADA"]["ROI_PREDICTOR_MODE"],
            "residual_trainable",
        )
        self.assertEqual(coco_base["SOLVER"]["WARMUP_ITERS"], 1000)
        base_mask = load_config(ROOT / "configs/coco/fsis/base_mask.yaml")
        self.assertFalse(base_mask["MODEL"]["EADA"]["FSIS_MASK_ONLY"])
        self.assertEqual(base_mask["SOLVER"]["BASE_LR"], 0.001)
        self.assertEqual(base_mask["SOLVER"]["MAX_ITER"], 10000)
        fsis = load_config(ROOT / "configs/coco/fsis/10shot.yaml")
        self.assertTrue(fsis["MODEL"]["MASK_ON"])
        self.assertTrue(fsis["MODEL"]["EADA"]["FSIS_MASK_ONLY"])
        self.assertTrue(fsis["MODEL"]["ROI_MASK_HEAD"]["CLS_AGNOSTIC_MASK"])

    def test_paper_support_split_inventory(self):
        for seed in (0, 1, 2):
            self.assertEqual(
                len(list((ROOT / f"splits/voc/seed{seed}").glob("box_*shot_*_train.txt"))),
                100,
            )
            self.assertEqual(
                len(list((ROOT / f"splits/coco/seed{seed}").glob("full_box_*shot_*_trainval.json"))),
                480,
            )

    def test_spdi_precedes_nms_and_saia_precedes_rpn(self):
        tree = ast.parse((ROOT / "eada/model.py").read_text())
        source = ast.unparse(tree)
        self.assertLess(
            source.index("calibrated.append"),
            source.index("fast_rcnn_inference("),
        )
        self.assertLess(
            source.index("self._features(images)"),
            source.index("self.proposal_generator("),
        )

    def test_readme_commands_have_no_patch_artifacts(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("+      ", readme)
        for name in (
            "INSTALL.md",
            "DATA.md",
            "MODEL_ZOO.md",
            "REPRODUCE.md",
            "expected/paper_results.json",
            "scripts/run_voc_cell.sh",
            "scripts/run_coco_cell.sh",
        ):
            self.assertIn(name, readme)


if __name__ == "__main__":
    unittest.main()
