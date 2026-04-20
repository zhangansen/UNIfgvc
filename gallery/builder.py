"""
gallery/builder.py — builds and checkpoints the multimodal gallery.

Stores 4 representations per support image:
    clip_vis  (512)   CLIP visual
    dino_vis  (768)   DINOv2 visual
    cdv_text  (512)   CLIP text on CDV paragraph
    fused     (1792)  concat + L2-norm of all three

All ablation modes are precomputed at build time so eval is fast.
"""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from models.encoders import (
    encode_clip_visual, encode_clip_text, encode_dino_visual, fuse
)
from models.cdv_captioner import full_cdv_caption


def _load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _get_confusable_refs(
    target_cls: int,
    class_prototypes: dict[int, torch.Tensor],
    support_set: dict[int, list[Path]],
    n: int = 3,
    query_img: Image.Image | None = None,
) -> tuple[list[Image.Image], list[str], list[str]]:
    """Return (ref_images, ref_class_names, ref_class_indices) for the n most similar classes."""
    if target_cls == -1:
        assert query_img is not None
        target_proto = encode_clip_visual(query_img)
        exclude = set()
    else:
        target_proto = class_prototypes[target_cls]
        exclude = {target_cls}

    sims = {
        c: (class_prototypes[c] @ target_proto).item()
        for c in class_prototypes if c not in exclude
    }
    top = sorted(sims, key=lambda c: -sims[c])[:n]
    return top  # list of class indices


def build_class_prototypes(
    support_set: dict[int, list[Path]],
) -> dict[int, torch.Tensor]:
    print("Building CLIP class prototypes for confusable-class lookup ...")
    prototypes = {}
    for cls_idx, paths in support_set.items():
        embs = torch.stack([encode_clip_visual(_load_pil(p)) for p in paths])
        prototypes[cls_idx] = F.normalize(embs.mean(0), dim=-1)
    print(f"  Done. {len(prototypes)} class prototypes computed.")
    return prototypes


def build_gallery(
    support_set: dict[int, list[Path]],
    class_names: list[str],
    out_dir: Path,
    k_shot: int = 4,
    n_ref_classes: int = 3,
    fast_cdv: bool = False,
    fusion_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
    checkpoint_every: int = 50,
) -> dict:
    """
    Build the multimodal gallery for all support images.

    Returns the gallery dict (also saved to out_dir/gallery_tensors.pt).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "gallery_checkpoint.pt"

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path)
        keys_clip_vis = ckpt["keys_clip_vis"]
        keys_dino_vis = ckpt["keys_dino_vis"]
        keys_cdv_text = ckpt["keys_cdv_text"]
        keys_fused    = ckpt["keys_fused"]
        labels_list   = ckpt["labels_list"]
        cdv_texts     = ckpt["cdv_texts"]
        done_set      = set(map(tuple, ckpt["done_set"]))
        print(f"Resumed checkpoint — {len(labels_list)} images already done.")
    else:
        keys_clip_vis, keys_dino_vis, keys_cdv_text, keys_fused = [], [], [], []
        labels_list, cdv_texts = [], []
        done_set = set()

    # ── Pre-compute class prototypes ──────────────────────────────────────────
    class_prototypes = build_class_prototypes(support_set)

    # ── Build loop ────────────────────────────────────────────────────────────
    all_items = [
        (cls_idx, img_path)
        for cls_idx in sorted(support_set.keys())
        for img_path in support_set[cls_idx]
    ]
    total   = len(all_items)
    t_start = time.time()

    for i, (cls_idx, img_path) in enumerate(all_items):
        item_key = (cls_idx, str(img_path))
        if item_key in done_set:
            continue

        t0    = time.time()
        image = _load_pil(img_path)

        # Encodings
        clip_v = encode_clip_visual(image)
        dino_v = encode_dino_visual(image)

        # CDV caption
        top_ref_cls = _get_confusable_refs(
            cls_idx, class_prototypes, support_set, n=n_ref_classes
        )
        ref_imgs  = [_load_pil(support_set[c][0]) for c in top_ref_cls]
        ref_names = [class_names[c] for c in top_ref_cls]
        cdv_p = full_cdv_caption(
            image, class_names[cls_idx], ref_imgs, ref_names, fast=fast_cdv
        )
        cdv_t  = encode_clip_text(cdv_p)
        fused  = fuse(clip_v, dino_v, cdv_t, weights=fusion_weights)

        keys_clip_vis.append(clip_v)
        keys_dino_vis.append(dino_v)
        keys_cdv_text.append(cdv_t)
        keys_fused.append(fused)
        labels_list.append(cls_idx)
        cdv_texts.append(cdv_p)
        done_set.add(item_key)

        done = len(labels_list)
        if done % checkpoint_every == 0 or done == total:
            elapsed = time.time() - t_start
            eta     = (elapsed / done) * (total - done) if done else 0
            print(
                f"  [{done}/{total}] cls={cls_idx} ({class_names[cls_idx][:25]}) | "
                f"{time.time()-t0:.1f}s/img | ETA {eta/60:.1f}min"
            )
            torch.save(dict(
                keys_clip_vis=keys_clip_vis, keys_dino_vis=keys_dino_vis,
                keys_cdv_text=keys_cdv_text, keys_fused=keys_fused,
                labels_list=labels_list,     cdv_texts=cdv_texts,
                done_set=[list(k) for k in done_set],
            ), ckpt_path)

    print(f"\nGallery built in {(time.time()-t_start)/60:.1f} min")

    # ── Stack & save ──────────────────────────────────────────────────────────
    gallery = {
        "keys_clip_vis": torch.stack(keys_clip_vis),   # (N*K, 512)
        "keys_dino_vis": torch.stack(keys_dino_vis),   # (N*K, 768)
        "keys_cdv_text": torch.stack(keys_cdv_text),   # (N*K, 512)
        "keys_fused":    torch.stack(keys_fused),      # (N*K, 1792)
        "labels":        torch.tensor(labels_list, dtype=torch.long),
    }
    torch.save(gallery, out_dir / "gallery_tensors.pt")
    with open(out_dir / "cdv_texts.json", "w") as f:
        json.dump(cdv_texts, f)

    print("Saved: gallery_tensors.pt, cdv_texts.json")
    for k, v in gallery.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")

    return gallery
