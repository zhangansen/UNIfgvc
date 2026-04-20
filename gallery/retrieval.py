"""
gallery/retrieval.py — cosine similarity retrieval with per-class mean aggregation.
"""

import torch


def retrieve(
    query_vec: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_labels: torch.Tensor,
    top_k: int = 5,
) -> list[int]:
    """
    Return top_k predicted class indices (descending confidence).

    query_vec      : (D,)   L2-normed
    gallery_feats  : (M, D) L2-normed
    gallery_labels : (M,)   int class indices
    """
    n_classes = int(gallery_labels.max().item()) + 1
    sims = gallery_feats @ query_vec          # (M,)

    class_scores = torch.zeros(n_classes)
    class_counts = torch.zeros(n_classes)
    for sim, lbl in zip(sims, gallery_labels):
        class_scores[lbl] += sim.item()
        class_counts[lbl] += 1
    class_scores /= class_counts.clamp(min=1)

    ranked = class_scores.argsort(descending=True).tolist()
    return ranked[:top_k]
