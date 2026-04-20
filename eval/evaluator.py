"""
eval/evaluator.py — runs all ablation modes and saves results.

Ablation modes (matches paper + your extension):
    clip_only   : CLIP visual only (512-dim)       ← paper baseline
    clip_cdv    : CLIP visual + CDV text (1024-dim) ← paper full method
    clip_dino   : CLIP + DINOv2 visual (1280-dim)  ← your visual extension
    full        : CLIP + DINOv2 + CDV text (1792)  ← your full method
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from models.encoders     import encode_clip_visual, encode_clip_text, encode_dino_visual, fuse
from models.cdv_captioner import full_cdv_caption
from gallery.retrieval   import retrieve
from gallery.builder     import _get_confusable_refs, build_class_prototypes


def _load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── Ablation gallery feature selectors ───────────────────────────────────────
def _gallery_feats_for_mode(gallery: dict, mode: str) -> torch.Tensor:
    if mode == "clip_only":
        return F.normalize(gallery["keys_clip_vis"], dim=-1)
    elif mode == "clip_cdv":
        raw = torch.cat([gallery["keys_clip_vis"], gallery["keys_cdv_text"]], dim=-1)
        return F.normalize(raw, dim=-1)
    elif mode == "clip_dino":
        raw = torch.cat([gallery["keys_clip_vis"], gallery["keys_dino_vis"]], dim=-1)
        return F.normalize(raw, dim=-1)
    elif mode == "full":
        return gallery["keys_fused"]
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")


def _query_vec_for_mode(
    image: Image.Image,
    mode: str,
    class_names: list[str],
    class_prototypes: dict,
    support_set: dict,
    n_ref_classes: int,
    fast_cdv: bool,
) -> torch.Tensor:
    clip_v = encode_clip_visual(image)

    if mode == "clip_only":
        return F.normalize(clip_v, dim=-1)

    dino_v = encode_dino_visual(image) if mode in ("clip_dino", "full") else None

    if mode in ("clip_cdv", "full"):
        top_ref_cls = _get_confusable_refs(
            -1, class_prototypes, support_set, n=n_ref_classes, query_img=image
        )
        ref_imgs  = [_load_pil(support_set[c][0]) for c in top_ref_cls]
        ref_names = [class_names[c] for c in top_ref_cls]
        cdv_p = full_cdv_caption(image, "", ref_imgs, ref_names, fast=fast_cdv)
        cdv_t = encode_clip_text(cdv_p)
    else:
        cdv_t = None

    return fuse(clip_v, dino_v, cdv_t)


ABLATION_MODES = ["clip_only", "clip_cdv", "clip_dino", "full"]
ABLATION_LABELS = {
    "clip_only": "CLIP visual only (512)",
    "clip_cdv":  "CLIP + CDV text (1024)",
    "clip_dino": "CLIP + DINOv2 visual (1280)",
    "full":      "CLIP + DINOv2 + CDV text (1792)",
}


def run_ablation(
    gallery: dict,
    test_set: dict[int, list[Path]],
    class_names: list[str],
    support_set: dict[int, list[Path]],
    n_ref_classes: int = 3,
    fast_cdv: bool = False,
    modes: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Runs all (or specified) ablation modes.
    Returns {mode: top1_accuracy}.
    """
    if modes is None:
        modes = ABLATION_MODES

    class_prototypes = build_class_prototypes(support_set)
    gallery_labels   = gallery["labels"]

    all_items = [
        (cls, p)
        for cls, paths in test_set.items()
        for p in paths
    ]

    results: dict[str, float] = {}

    for mode in modes:
        print(f"\n── Ablation: {ABLATION_LABELS[mode]} ──")
        gal_feats = _gallery_feats_for_mode(gallery, mode)
        correct, total = 0, 0
        t_start = time.time()

        for i, (true_cls, img_path) in enumerate(all_items):
            image = _load_pil(img_path)
            q_vec = _query_vec_for_mode(
                image, mode, class_names, class_prototypes,
                support_set, n_ref_classes, fast_cdv
            )
            preds = retrieve(q_vec, gal_feats, gallery_labels, top_k=1)
            correct += int(preds[0] == true_cls)
            total   += 1

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                eta     = elapsed / (i + 1) * (len(all_items) - i - 1)
                print(
                    f"  [{i+1}/{len(all_items)}] acc={correct/total:.3f} "
                    f"| ETA {eta/60:.1f}min"
                )

        acc = correct / total if total else 0.0
        results[mode] = acc
        print(f"  Top-1 accuracy: {acc:.4f}  ({correct}/{total})")

    return results
