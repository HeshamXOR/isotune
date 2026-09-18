"""End-to-end smoke test: does IsoLatencySearchCV actually deliver a model inside its budget?

Not a benchmark. This checks the class runs, respects the budget it was given, and reports a
re-measured latency rather than a predicted one.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from isotune import IsoLatencySearchCV

X, y = make_classification(n_samples=6000, n_features=25, n_informative=12,
                           flip_y=0.05, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

space = {
    "max_depth": [2, 3, 4, 6, 8, 10],
    "learning_rate": [0.03, 0.06, 0.1, 0.2],
    "min_child_weight": [0.5, 1.0, 5.0, 20.0],
    "subsample": [0.7, 0.85, 1.0],
}

results = []
for budget_ms in (2.0, 5.0, 15.0):
    search = IsoLatencySearchCV(
        XGBClassifier(tree_method="hist", n_jobs=1, verbosity=0, eval_metric="logloss"),
        space,
        max_latency=budget_ms / 1000.0,
        batch_size=1000,
        n_candidates=12,
        max_trees=400,
        probe_trees=16,
        random_state=0,
        verbose=0,
    ).fit(Xtr, ytr)

    auc = roc_auc_score(yte, search.predict_proba(Xte)[:, 1])
    print(f"budget {budget_ms:5.1f} ms | depth {search.best_n_trees_:4d} trees "
          f"| max_depth {search.best_params_['max_depth']:2d} "
          f"| measured {search.measured_latency_*1e3:6.3f} ms "
          f"| within budget: {search.latency_ok_} "
          f"| excluded {search.excluded_:2d} | test AUC {auc:.4f}")

    assert search.best_n_trees_ >= 1
    assert search.measured_latency_ > 0
    assert search.best_n_trees_ <= 400
    # The promise of the library. If this fails the delivered model breaches the budget it was
    # selected under, which is the one thing isotune exists to prevent.
    assert search.latency_ok_, (
        f"delivered {search.measured_latency_*1e3:.3f}ms against a "
        f"{budget_ms:.1f}ms budget")
    results.append((budget_ms, search.best_n_trees_, auc))

# Deeper budgets must buy at least as many trees; a non-monotone depth means the allocator is
# not responding to the budget at all.
depths = [d for _, d, _ in results]
assert depths == sorted(depths), f"depth should grow with budget, got {depths}"

print("\nsmoke test OK")
