import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def expand_state(state, base_names, all_names):
    result = dict(state)
    positions = [all_names.index(name) for name in base_names]
    for head in ("cls_score", "bbox_pred"):
        block = 1 if head == "cls_score" else 4
        rows = len(all_names) + 1 if head == "cls_score" else 4 * len(all_names)
        for suffix in ("weight", "bias"):
            key = "roi_heads.box_predictor." + head + "." + suffix
            old = state[key]
            expected = len(base_names) + 1 if head == "cls_score" else 4 * len(base_names)
            if old.shape[0] != expected:
                raise ValueError("Unexpected source shape for " + key)
            new = old.new_zeros((rows,) + old.shape[1:])
            if suffix == "weight":
                torch.nn.init.normal_(new, std=0.01 if head == "cls_score" else 0.001)
            for source_index, target_index in enumerate(positions):
                new[target_index * block:(target_index + 1) * block] = old[source_index * block:(source_index + 1) * block]
            if head == "cls_score":
                new[-1] = old[-1]
            result[key] = new
    return result


def main():
    parser = argparse.ArgumentParser(description="Expand trusted base detector checkpoints by class name")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from eada.data.builtin_meta import _get_builtin_metadata
    meta = _get_builtin_metadata(args.dataset + "_fewshot")
    base_names, all_names = meta["base_classes"], meta["thing_classes"]
    if args.dataset == "voc":
        base_names, all_names = base_names[args.split], all_names[args.split]
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.source, map_location="cpu")
    expanded = expand_state(checkpoint["model"], base_names, all_names)
    Path(args.destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": expanded}, args.destination)


if __name__ == "__main__":
    main()
