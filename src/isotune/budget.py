"""Latency budgeting: turn a millisecond budget into a per-configuration tree budget.

The whole library rests on one measured fact. For a fitted gradient-boosted model, prediction
latency over a fixed batch is close to affine in the number of trees traversed,

    latency(k) ~= p + q * k

and `q` -- the per-tree traversal cost -- varies by **7x to 38x across hyperparameter
configurations** of the same search space (measured on 23 tabular datasets). So a fixed
millisecond budget buys one configuration many more trees than another, and ranking
configurations at an equal number of trees is not ranking them at equal latency.

Everything in this module is pure arithmetic on measured coefficients. Nothing here fits a model.
"""

from __future__ import annotations

import numpy as np

__all__ = ["affordable_trees", "budget_from_reference", "is_feasible", "latency_of"]


def latency_of(p: float, q: float, n_trees: int) -> float:
    """Predicted latency in seconds for `n_trees` of a configuration with profile (p, q)."""
    return float(p) + float(q) * int(n_trees)


def affordable_trees(p, q, budget_seconds: float, grid=None, max_trees: int | None = None):
    """Deepest tree count each configuration can serve inside `budget_seconds`.

    Returns an integer array; ``0`` marks a configuration that cannot afford even one tree and
    must be excluded from the search entirely.

    Rounding is always **down**. If `grid` is given, the result is floored onto it, so a returned
    depth is always one you can actually serve. Rounding up would hand back a model that violates
    the budget it was selected under, which defeats the point of the library.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("latency profiles must be finite")
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(q > 0, (budget_seconds - p) / np.where(q > 0, q, 1.0), np.inf)
    raw = np.where(np.isfinite(raw), raw, float(max_trees or 0))
    out = np.floor(np.maximum(raw, 0.0)).astype(int)

    if max_trees is not None:
        out = np.minimum(out, int(max_trees))
    if grid is not None:
        g = np.sort(np.asarray(grid, dtype=int))
        idx = np.searchsorted(g, out, side="right") - 1
        out = np.where(idx >= 0, g[np.clip(idx, 0, len(g) - 1)], 0)
    # A configuration whose fixed overhead alone exceeds the budget affords nothing.
    return np.where(p > budget_seconds, 0, out).astype(int)


def budget_from_reference(p, q, reference_trees: int) -> float:
    """A latency budget defined so the MEDIAN configuration affords exactly `reference_trees`.

    Useful for benchmarking and for choosing a budget when you have no externally imposed SLA:
    it makes the budget comparable across datasets and hardware. Under this definition any
    advantage a latency-aware search shows comes from how it treats configurations away from the
    median, not from being handed a bigger budget than its baseline.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.median(p + q * int(reference_trees)))


def is_feasible(p: float, q: float, n_trees: int, budget_seconds: float, tol: float = 1e-12) -> bool:
    """Would this configuration at this depth meet the budget, by the fitted profile?

    The profile is a model of measured latency, not a guarantee. Verify the delivered model with
    `isotune.profile.measure_latency` before relying on it for anything with an SLA attached.
    """
    return latency_of(p, q, n_trees) <= float(budget_seconds) + tol
