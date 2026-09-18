"""isotune -- hyperparameter search whose deliverable meets an inference-latency budget.

    from isotune import IsoLatencySearchCV

    search = IsoLatencySearchCV(
        XGBClassifier(tree_method="hist"),
        {"max_depth": [2, 4, 6, 8, 10], "learning_rate": [0.03, 0.1, 0.3]},
        max_latency=0.005,        # 5 ms per batch
        batch_size=1000,
    ).fit(X, y)

    search.best_n_trees_, search.measured_latency_

Why this is not just "tune, then truncate": per-tree inference cost varies 7x-38x across
configurations of the same search space, so a millisecond budget buys different depths to
different configurations, and comparing them at equal trees compares them at unequal latency.
"""

from .budget import affordable_trees, budget_from_reference, is_feasible, latency_of
from .profile import DEFAULT_BATCH, measure_latency, profile_booster, time_callables
from .search import IsoLatencySearchCV

__version__ = "0.1.0"

__all__ = [
    "IsoLatencySearchCV",
    "affordable_trees",
    "budget_from_reference",
    "is_feasible",
    "latency_of",
    "measure_latency",
    "profile_booster",
    "time_callables",
    "DEFAULT_BATCH",
    "__version__",
]
