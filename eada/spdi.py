import math
import torch
import torch.nn.functional as functional
from PIL import Image


PROMPTS = ("a photo of a {}", "a clean photo of a {}")


def fuse_novel(
    detector,
    clip_probabilities,
    novel_ids,
    weight=0.7,
    mode="absolute",
):
    """Apply SPDI to novel coordinates and leave base/background unchanged.

    ``absolute`` is the paper EADA formulation: detector and CLIP novel-class
    probabilities are directly blended. ``normalized`` is the Table-6 control:
    preserve the detector's total novel probability mass and only change the
    conditional distribution inside the novel subset.
    """
    result = detector.clone()
    if not novel_ids or detector.numel() == 0:
        return result
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("SPDI detector weight must be in [0, 1]")
    if mode == "absolute":
        result[:, novel_ids] = (
            float(weight) * detector[:, novel_ids]
            + (1.0 - float(weight)) * clip_probabilities[:, novel_ids]
        )
    elif mode == "normalized":
        eps = torch.finfo(detector.dtype).eps
        detector_novel = detector[:, novel_ids]
        detector_mass = detector_novel.sum(dim=-1, keepdim=True)
        detector_conditional = detector_novel / detector_mass.clamp_min(eps)
        clip_novel = clip_probabilities[:, novel_ids]
        clip_conditional = clip_novel / clip_novel.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        mixed = (
            float(weight) * detector_conditional
            + (1.0 - float(weight)) * clip_conditional
        )
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp_min(eps)
        result[:, novel_ids] = detector_mass * mixed
    else:
        raise ValueError("SPDI fusion mode must be absolute or normalized")
    return result


def select_proposals(scores, topk=100):
    if topk <= 0:
        return torch.arange(len(scores), device=scores.device)
    return scores[:, :-1].amax(dim=1).topk(min(topk, len(scores))).indices


class SPDI:
    """Semantic-Prior Decision Integration with frozen OpenAI CLIP weights."""

    def __init__(
        self,
        class_names,
        novel_ids,
        device,
        checkpoint="",
        model_name="ViT-B-16",
        pretrained="openai",
        prompts=PROMPTS,
        topk=100,
        temperature=0.01,
        detector_weight=0.7,
        fusion_mode="absolute",
        batch_size=64,
    ):
        import open_clip

        if model_name != "ViT-B-16":
            raise ValueError("EADA SPDI requires CLIP ViT-B/16")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=checkpoint or pretrained,
        )
        self.model = self.model.to(device)
        self.model.eval().requires_grad_(False)
        self.novel_ids = list(novel_ids)
        self.device = device
        self.topk = int(topk)
        self.temperature = max(float(temperature), 1e-6)
        self.detector_weight = float(detector_weight)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"absolute", "normalized"}:
            raise ValueError("SPDI fusion mode must be absolute or normalized")
        self.batch_size = max(int(batch_size), 1)
        self.prompts = tuple(prompts)
        with torch.no_grad():
            tokens = open_clip.tokenize(
                [
                    template.format(name)
                    for name in class_names
                    for template in self.prompts
                ]
            ).to(device)
            text = functional.normalize(self.model.encode_text(tokens).float(), dim=-1)
            self.text = functional.normalize(
                text.reshape(len(class_names), len(self.prompts), -1).mean(1),
                dim=-1,
            )

    @torch.no_grad()
    def __call__(self, image, boxes, scores):
        if not len(scores) or not self.novel_ids:
            return scores
        selected = select_proposals(scores, self.topk)
        if boxes.shape[1] == 4:
            selected_boxes = boxes[selected]
        else:
            classes = scores[selected, :-1].argmax(1)
            selected_boxes = boxes.reshape(len(scores), -1, 4)[selected, classes]
        height, width = image.shape[-2:]
        pixels = (
            image.detach().clamp(0, 255).round().to(torch.uint8)
            .permute(1, 2, 0).cpu().numpy()
        )
        pil_image = Image.fromarray(pixels)
        crops, valid = [], []
        for index, box in zip(selected.tolist(), selected_boxes.cpu()):
            left, top, right, bottom = box.tolist()
            if not all(math.isfinite(value) for value in (left, top, right, bottom)):
                continue
            left = max(0, int(math.floor(left)))
            top = max(0, int(math.floor(top)))
            right = min(width, int(math.ceil(right)))
            bottom = min(height, int(math.ceil(bottom)))
            if right <= left or bottom <= top:
                continue
            crops.append(self.preprocess(pil_image.crop((left, top, right, bottom))))
            valid.append(index)
        if not crops:
            return scores
        probabilities = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[start:start + self.batch_size]).to(self.device)
            visual = functional.normalize(self.model.encode_image(batch).float(), dim=-1)
            similarities = visual @ self.text.T / self.temperature
            probabilities.append(similarities.softmax(dim=-1))
        result = scores.clone()
        result[valid] = fuse_novel(
            scores[valid],
            torch.cat(probabilities).to(scores),
            self.novel_ids,
            self.detector_weight,
            self.fusion_mode,
        )
        return result

    @torch.no_grad()
    def rescore_instances(self, image, instances):
        """Matched post-NMS SPDI placement control used in Table 6.

        Only the score of the class that already survived detector NMS is
        updated. Class labels and boxes are intentionally unchanged because
        competing hypotheses have already been removed at this placement.
        """
        if len(instances) == 0 or not self.novel_ids:
            return instances
        count = min(self.topk if self.topk > 0 else len(instances), len(instances))
        selected = instances.scores.topk(count).indices
        boxes = instances.pred_boxes.tensor[selected]
        height, width = image.shape[-2:]
        pixels = (
            image.detach().clamp(0, 255).round().to(torch.uint8)
            .permute(1, 2, 0).cpu().numpy()
        )
        pil_image = Image.fromarray(pixels)
        crops, valid = [], []
        for local_index, box in enumerate(boxes.cpu()):
            left, top, right, bottom = box.tolist()
            if not all(math.isfinite(value) for value in (left, top, right, bottom)):
                continue
            left = max(0, int(math.floor(left)))
            top = max(0, int(math.floor(top)))
            right = min(width, int(math.ceil(right)))
            bottom = min(height, int(math.ceil(bottom)))
            if right <= left or bottom <= top:
                continue
            crops.append(self.preprocess(pil_image.crop((left, top, right, bottom))))
            valid.append(local_index)
        if not crops:
            return instances
        probabilities = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[start:start + self.batch_size]).to(self.device)
            visual = functional.normalize(self.model.encode_image(batch).float(), dim=-1)
            similarities = visual @ self.text.T / self.temperature
            probabilities.append(similarities.softmax(dim=-1))
        semantic = torch.cat(probabilities).to(instances.scores)
        novel = set(self.novel_ids)
        for semantic_row, local_index in zip(semantic, valid):
            global_index = int(selected[local_index])
            class_id = int(instances.pred_classes[global_index])
            if class_id not in novel:
                continue
            instances.scores[global_index] = (
                self.detector_weight * instances.scores[global_index]
                + (1.0 - self.detector_weight) * semantic_row[class_id]
            )
        return instances
