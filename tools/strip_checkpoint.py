#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Write a model-only reviewer checkpoint")
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, destination)
    print(destination)


if __name__ == "__main__":
    main()

