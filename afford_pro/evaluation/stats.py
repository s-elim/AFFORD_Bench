"""Success-rate statistics: Wilson intervals over episodes and task-clustered bootstrap intervals.

Episodes of one task share a scene and an oracle or policy, so they are not independent; the Wilson interval over
all episodes of a domain is too narrow. `cluster_bootstrap_ci` resamples tasks (with all of their episodes) instead.
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def cluster_bootstrap_ci(per_task_successes: list[list[bool]], n_boot: int = 10_000, seed: int = 0,
                         level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the pooled success rate when tasks (clusters) are resampled with replacement."""
    clusters = [np.asarray(c, dtype=float) for c in per_task_successes if len(c)]
    if not clusters:
        return float("nan"), float("nan")
    sums = np.array([c.sum() for c in clusters])
    sizes = np.array([len(c) for c in clusters])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    rates = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(rates, alpha)), float(np.quantile(rates, 1.0 - alpha))


def summarize(per_task_successes: list[list[bool]]) -> dict[str, float | int]:
    flat = [s for c in per_task_successes for s in c]
    k, n = int(sum(flat)), len(flat)
    lo, hi = wilson_ci(k, n)
    clo, chi = cluster_bootstrap_ci(per_task_successes)
    return {"successes": k, "episodes": n, "tasks": len(per_task_successes), "rate": k / n if n else float("nan"),
            "wilson95": [lo, hi], "task_cluster_bootstrap95": [clo, chi]}
