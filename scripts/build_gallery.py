#!/usr/bin/env python
"""
scripts/build_gallery.py — build the multimodal gallery for a dataset.

Usage:
    python scripts/build_gallery.py \
        --dataset cub \
        --data_root /path/to/cub \
        --out_dir results/cub \
        --k_shot 4 \
        --fast_cdv   # omit for full 3-stage CDV
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils     import setup_hf_token
from datasets  import get_dataset
from models    import load_encoders, load_qwen
from gallery   import build_gallery


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      required=True,
                   choices=["cub", "cars", "aircraft", "pets", "dogs", "food101"])
    p.add_argument("--data_root",    required=True,  help="Path to dataset root")
    p.add_argument("--out_dir",      required=True,  help="Where to save gallery files")
    p.add_argument("--k_shot",       type=int, default=4)
    p.add_argument("--n_ref_classes",type=int, default=3)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--fast_cdv",     action="store_true",
                   help="Use single-prompt CDV (faster, less accurate)")
    p.add_argument("--checkpoint_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    setup_hf_token()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset, args.data_root)
    class_names, train_dict, _ = dataset.load()
    print(f"  {len(class_names)} classes, {sum(len(v) for v in train_dict.values())} train images")

    # Sample support set
    rng = random.Random(args.seed)
    support_set = {
        cls: rng.sample(paths, min(args.k_shot, len(paths)))
        for cls, paths in train_dict.items()
    }
    print(f"  Support set: {sum(len(v) for v in support_set.values())} images ({args.k_shot}-shot)")

    # Load models
    load_encoders()
    load_qwen()

    # Build gallery
    out_dir = Path(args.out_dir)
    build_gallery(
        support_set=support_set,
        class_names=class_names,
        out_dir=out_dir,
        k_shot=args.k_shot,
        n_ref_classes=args.n_ref_classes,
        fast_cdv=args.fast_cdv,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"\nGallery saved to: {out_dir}")


if __name__ == "__main__":
    main()
