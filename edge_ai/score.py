"""Turn measurements into the contest's 100-point shape.

Judges rescore on their own computer. This local score is only how we choose
which candidate to submit. It follows the bands in the pre-competition packet:

- Accuracy: keep at least 97% of the starting accuracy, on every land type.
- Compression: shrink BOTH the parameter count and the file.
- Speed, memory, and energy: lower is better. Those are ratios against our
  own starting model, because every laptop is different.
"""

from __future__ import annotations


def accuracy_points(accuracy: float, baseline: float, worst: float, baseline_worst: float) -> float:
    if baseline <= 0:
        return 0.0
    kept = accuracy / baseline
    worst_drop = baseline_worst - worst
    if kept >= 0.97 and worst_drop <= 0.05:
        return 24.0
    if kept >= 0.97:
        return 21.0
    if kept >= 0.90:
        return 17.0
    if kept >= 0.75:
        return 12.0
    return 4.0


def _band(ratio: float) -> float:
    if ratio >= 10:
        return 19.0
    if ratio >= 5:
        return 15.0
    if ratio >= 2:
        return 10.0
    if ratio >= 1.3:
        return 6.0
    return 3.0


def compression_points(param_ratio: float, size_ratio: float) -> float:
    """Half the points come from fewer parameters, half from a smaller file.

    The public sheet's top band still needs both to shrink a lot. Averaging
    the two bands means an 8-bit file gets credit even when the channel count
    stays the same, and channel cuts get credit even before the 8-bit save.
    """
    return 0.5 * _band(param_ratio) + 0.5 * _band(size_ratio)


def lower_is_better_points(ratio: float) -> float:
    """ratio = baseline / new. 2.0 means the new one uses half as much."""
    if ratio >= 8:
        return 19.0
    if ratio >= 4:
        return 16.0
    if ratio >= 2:
        return 13.0
    if ratio >= 1.25:
        return 9.0
    return 4.0


def score_candidate(candidate: dict, baseline: dict) -> float:
    accuracy = accuracy_points(
        candidate["accuracy"],
        baseline["accuracy"],
        candidate["worst_class_accuracy"],
        baseline["worst_class_accuracy"],
    )
    params_ratio = baseline["params_million"] / max(candidate["params_million"], 1e-9)
    size_ratio = baseline["size_mb"] / max(candidate["size_mb"], 1e-9)
    compression = compression_points(params_ratio, size_ratio)
    speed = lower_is_better_points(baseline["latency_ms"] / max(candidate["latency_ms"], 1e-9))
    memory = lower_is_better_points(baseline["peak_ram_mb"] / max(candidate["peak_ram_mb"], 1e-9))
    energy = lower_is_better_points(
        baseline["joules_per_inference"] / max(candidate["joules_per_inference"], 1e-12)
    )
    # Speed and memory share the 20-point resource bucket.
    total = accuracy + compression + 0.5 * (speed + memory) + energy
    candidate["score"] = round(total, 2)
    candidate["params_ratio"] = round(params_ratio, 3)
    candidate["size_ratio"] = round(size_ratio, 3)
    candidate["accuracy_kept"] = round(candidate["accuracy"] / max(baseline["accuracy"], 1e-9), 4)
    return candidate["score"]


def pick_winner(candidates: list[dict], baseline: dict) -> dict:
    """Keep accuracy, then the worst land type, then the smaller file.

    Demo: 128 pixels had a higher speed score and was picked first. It also
    made River the worst class. A candidate is now set aside when its worst
    land type falls below the starting model's worst land type.
    """
    worst_floor = baseline.get("worst_class_accuracy", 0)

    def holds_worst_class(item) -> bool:
        return item.get("accuracy_kept", 0) >= 0.97 and item.get("worst_class_accuracy", 0) + 1e-6 >= worst_floor

    safe = [item for item in candidates if holds_worst_class(item)]
    if not safe:
        safe = [item for item in candidates if item.get("accuracy_kept", 0) >= 0.97]
    pool = safe or candidates
    # Same score: smaller file, then fewer weights, then the stronger weak class.
    return max(
        pool,
        key=lambda item: (
            item.get("score", 0),
            item.get("size_ratio", 0),
            item.get("params_ratio", 0),
            item.get("worst_class_accuracy", 0),
        ),
    )
