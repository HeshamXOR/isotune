"""Measure, do not predict: fitting latency_c(k) ~= p + q*k for a booster.

WHY MEASURE. The obvious shortcut is to predict per-tree cost from the hyperparameters --
`max_depth` looks like it should determine traversal cost. It does not, reliably. Across 23
tabular datasets the pooled Spearman correlation between `max_depth` and measured per-tree cost
is only **+0.588**, and the best predictor changes by dataset: `min_child_weight` carries it on
one (-0.830) and is nearly useless on another (-0.127). Any library that priced latency from
hyperparameters alone would mis-allocate.

So `isotune` measures `q` directly, once per configuration, after that configuration has been
fitted to any depth. The cost is one short timing run; the benefit is a latency model that is
right on your hardware, at your batch size, for your data.

METHODOLOGY NOTES, both of which change the numbers materially:

* **Batch size matters.** Below roughly 512 rows, fixed per-call overhead dominates and swamps
  the per-tree term this module is trying to resolve. Profile at the batch size you will actually
  serve. `DEFAULT_BATCH = 2048` is a reasonable default for offline scoring, not for single-row
  online serving -- for that, profile at `batch_size=1`.
* **Interleave the repeats.** Timing each prefix to completion in turn attributes any thermal or
  scheduler drift to whichever prefix ran last. Repeats are interleaved in a fresh random order,
  and the median across repeats is taken.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

__all__ = ["DEFAULT_BATCH", "measure_latency", "time_callables", "profile_booster"]

DEFAULT_BATCH = 2048


def time_callables(functions: dict[str, Callable[[], object]], repeats: int = 5,
                   inner: int = 2, seed: int = 0) -> dict[str, float]:
    """Warm up once, then interleave repeated timings; return the median seconds per call."""
    if repeats < 1 or inner < 1 or not functions:
        raise ValueError("need at least one function and positive repeat counts")
    for fn in functions.values():
        fn()
    rng = np.random.default_rng(seed)
    keys = list(functions)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(repeats):
        for i in rng.permutation(len(keys)):
            k = keys[int(i)]
            start = time.perf_counter()
            for _ in range(inner):
                functions[k]()
            samples[k].append((time.perf_counter() - start) / inner)
    return {k: float(np.median(v)) for k, v in samples.items()}


def measure_latency(predict_at: Callable[[int], object], prefixes: Sequence[int],
                    repeats: int = 5, inner: int = 2, seed: int = 0) -> tuple[float, float]:
    """Fit ``latency(k) ~= p + q*k`` from timings of `predict_at(k)` over `prefixes`.

    `predict_at(k)` must run a prediction using exactly the first `k` trees, on a batch that is
    already prepared -- batch construction should happen outside, or it lands in `p` and the
    per-tree slope is still recovered correctly but `p` stops meaning what you think.

    Returns ``(p_seconds, q_seconds_per_tree)``. A non-positive `q` means the timing was too noisy
    to resolve a per-tree term at this batch size; callers should treat that configuration as
    unprofiled rather than trusting the fit.
    """
    ks = np.asarray(sorted(set(int(k) for k in prefixes)), dtype=float)
    if len(ks) < 2:
        raise ValueError("need at least two distinct prefixes to fit a slope")
    timed = time_callables({str(int(k)): (lambda k=int(k): predict_at(k)) for k in ks},
                           repeats=repeats, inner=inner, seed=seed)
    ys = np.array([timed[str(int(k))] for k in ks], dtype=float)
    q, p = np.polyfit(ks, ys, 1)
    return float(p), float(q)


def profile_booster(model, X, prefixes: Sequence[int] | None = None,
                    repeats: int = 5, inner: int = 2, seed: int = 0) -> tuple[float, float]:
    """Profile a fitted XGBoost or LightGBM model over a batch `X`.

    Dispatches on whichever prefix-scoring API the model exposes, so the same call works for
    `xgboost.XGBClassifier`, a raw `xgboost.Booster`, and `lightgbm.LGBMClassifier`. Prefix
    scoring is what makes this cheap: one fitted model answers every depth, so profiling costs a
    handful of predictions rather than a handful of fits.
    """
    X = np.asarray(X)
    n = _n_trees(model)
    if prefixes is None:
        top = max(n, 2)
        prefixes = sorted({max(1, int(round(top * f))) for f in (0.125, 0.25, 0.5, 1.0)})
    fn = _prefix_predictor(model, X)
    return measure_latency(fn, prefixes, repeats=repeats, inner=inner, seed=seed)


def _n_trees(model) -> int:
    for attr in ("num_boosted_rounds",):
        if hasattr(model, attr):
            return int(getattr(model, attr)())
    for attr in ("n_estimators", "n_estimators_", "num_trees"):
        v = getattr(model, attr, None)
        if callable(v):
            return int(v())
        if v is not None:
            return int(v)
    booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
    if booster is not None:
        return _n_trees(booster)
    raise TypeError(f"cannot determine tree count for {type(model).__name__}")


def _prefix_predictor(model, X) -> Callable[[int], object]:
    """Return f(k) running a prediction restricted to the first k trees."""
    import_err = None
    # xgboost sklearn API
    if hasattr(model, "predict_proba"):
        try:
            model.predict_proba(X[:2], iteration_range=(0, 1))
            return lambda k: model.predict_proba(X, iteration_range=(0, int(k)))
        except TypeError as exc:
            import_err = exc
        try:  # lightgbm sklearn API
            model.predict_proba(X[:2], num_iteration=1)
            return lambda k: model.predict_proba(X, num_iteration=int(k))
        except TypeError as exc:
            import_err = exc
    # raw xgboost Booster
    if hasattr(model, "predict") and hasattr(model, "num_boosted_rounds"):
        import xgboost as xgb
        dm = xgb.DMatrix(X)
        return lambda k: model.predict(dm, iteration_range=(0, int(k)))
    # raw lightgbm Booster
    if hasattr(model, "predict") and hasattr(model, "num_trees"):
        return lambda k: model.predict(X, num_iteration=int(k))
    raise TypeError(
        f"{type(model).__name__} does not expose a prefix-scoring API "
        f"(tried iteration_range= and num_iteration=); last error: {import_err}"
    )
