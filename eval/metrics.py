"""
eval/metrics.py — top-k accuracy helpers.
"""


def topk_accuracy(correct_at_k: list[bool]) -> float:
    return sum(correct_at_k) / len(correct_at_k) if correct_at_k else 0.0


def format_results_table(
    dataset_name: str,
    results: dict[str, float],
    k_shot: int,
) -> str:
    from eval.evaluator import ABLATION_LABELS
    lines = [
        f"\n{'='*60}",
        f"Dataset: {dataset_name.upper()}  |  {k_shot}-shot",
        f"{'='*60}",
        f"{'Mode':<35} {'Top-1 Acc':>10}",
        f"{'-'*50}",
    ]
    best = max(results.values()) if results else 0
    for mode, acc in results.items():
        label  = ABLATION_LABELS.get(mode, mode)
        marker = " ◀ best" if acc == best else ""
        lines.append(f"{label:<35} {acc:>10.4f}{marker}")
    lines.append("=" * 60)
    return "\n".join(lines)
