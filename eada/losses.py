import torch
import torch.nn.functional as functional


def dc_logits(logits, labels, counts, observed_classes):
    """Gate only background RoIs; unobserved logits become zero, not -inf."""
    if len(counts) != len(observed_classes) or sum(counts) != len(logits):
        raise ValueError("DC image/proposal alignment mismatch")
    background = logits.shape[1] - 1
    mask = torch.ones_like(logits)
    offset = 0
    for count, observed in zip(counts, observed_classes):
        known = sorted(set(int(value) for value in observed) | {background})
        if any(value < 0 or value > background for value in known):
            raise ValueError("DC class index out of range")
        rows = torch.arange(offset, offset + count, device=logits.device)
        rows = rows[labels[rows] == background]
        mask[rows] = 0
        mask[rows[:, None], known] = 1
        offset += count
    return logits * mask


def dc_loss(logits, labels, counts, observed_classes):
    if logits.numel() == 0:
        return logits.sum() * 0
    return functional.cross_entropy(dc_logits(logits, labels, counts, observed_classes), labels)
