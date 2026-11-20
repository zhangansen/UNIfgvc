#!/usr/bin/env python
"""
scripts/evaluate.py — run ablation evaluation and save results as CSV + JSON.

Usage:
    python scripts/evaluate.py \
        --dataset cub \
        --data_root /path/to/cub \
        --gallery_dir results/cub \
        --out_dir results/cub \
        --k_shot 4 \
        --n_test_per_class 3

Results are written to:
    results/cub/eval_results.json
    results/cub/eval_results.csv   (append-friendly, one row per dataset+mode)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils     import setup_hf_token, setup_vlm_api_key
from datasets  import get_dataset
from models    import load_encoders, load_qwen
from gallery   import build_gallery
from eval      import run_ablation, format_results_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",         required=True,
                   choices=["cub", "cars", "aircraft", "pets", "dogs", "food101"])
    p.add_argument("--data_root",       required=True)
    p.add_argument("--gallery_dir",     required=True,
                   help="Directory containing gallery_tensors.pt")
    p.add_argument("--out_dir",         required=True,
                   help="Where to write eval_results.json / .csv")
    p.add_argument("--k_shot",          type=int, default=4)
    p.add_argument("--n_test_per_class",type=int, default=3)
    p.add_argument("--n_ref_classes",   type=int, default=3)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--fast_cdv",        action="store_true")
    p.add_argument("--modes",           nargs="+",
                   default=["clip_only", "clip_cdv", "clip_dino", "full"],
                   help="Ablation modes to run")
    return p.parse_args()


def main():
    args = parse_args()
    setup_hf_token()
    setup_vlm_api_key()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset, args.data_root)
    class_names, train_dict, test_dict = dataset.load()

    rng = random.Random(args.seed)
    support_set = {
        cls: rng.sample(paths, min(args.k_shot, len(paths)))
        for cls, paths in train_dict.items()
    }
    test_set = {
        cls: paths[:args.n_test_per_class]
        for cls, paths in test_dict.items()
    }

    total_test = sum(len(v) for v in test_set.values())
    print(f"  {len(class_names)} classes | "
          f"{sum(len(v) for v in support_set.values())} support | "
          f"{total_test} test images")

    # ── Load models ───────────────────────────────────────────────────────────
    load_encoders()
    load_qwen()

    # ── Load gallery ──────────────────────────────────────────────────────────
    gallery_path = Path(args.gallery_dir) / "gallery_tensors.pt"
    if not gallery_path.exists():
        raise FileNotFoundError(
            f"Gallery not found at {gallery_path}. "
            "Run scripts/build_gallery.py first."
        )
    gallery = torch.load(gallery_path)
    print(f"Gallery loaded: {gallery['keys_fused'].shape}")

    # ── Run ablation ──────────────────────────────────────────────────────────
    results = run_ablation(
        gallery=gallery,
        test_set=test_set,
        class_names=class_names,
        support_set=support_set,
        n_ref_classes=args.n_ref_classes,
        fast_cdv=args.fast_cdv,
        modes=args.modes,
        verbose=True,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print(format_results_table(args.dataset, results, args.k_shot))

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "dataset":          args.dataset,
        "k_shot":           args.k_shot,
        "n_test_per_class": args.n_test_per_class,
        "fast_cdv":         args.fast_cdv,
        "seed":             args.seed,
        "results":          results,
    }
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")

    # CSV (append-friendly — one row per dataset + mode)
    csv_path = out_dir / "eval_results.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "k_shot", "n_test_per_class",
                             "fast_cdv", "seed", "mode", "top1_acc"])
        for mode, acc in results.items():
            writer.writerow([
                args.dataset, args.k_shot, args.n_test_per_class,
                args.fast_cdv, args.seed, mode, f"{acc:.4f}",
            ])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
