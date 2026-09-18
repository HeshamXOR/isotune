<div align="center">

# isotune

### Hyperparameter search that respects your inference latency budget

**Your model won the tuning run. Then it was too slow to ship.**<br>
`isotune` makes the latency budget part of the search instead of a regret afterwards.

[![CI](https://github.com/HeshamXOR/isotune/actions/workflows/ci.yml/badge.svg)](https://github.com/HeshamXOR/isotune/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://pypi.org/project/isotune/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-supported-EC4E20)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-supported-9ACD32)](https://lightgbm.readthedocs.io/)
[![Code style](https://img.shields.io/badge/scikit--learn-API%20compatible-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

---

```python
from isotune import IsoLatencySearchCV
from xgboost import XGBClassifier

search = IsoLatencySearchCV(
    XGBClassifier(tree_method="hist"),
    {"max_depth": [2, 3, 4, 6, 8, 10], "learning_rate": [0.03, 0.1, 0.2]},
    max_latency=0.005,      # 5 ms per batch — for the model you actually deploy
    batch_size=1000,
).fit(X, y)

search.best_n_trees_       # 96      — the depth it can afford, not the depth it liked
search.measured_latency_   # 0.00397 — measured on your hardware, not predicted
search.latency_ok_         # True    — verified, with margin
```

---

## The problem

You tune to convergence, the winner is too slow, so you truncate it until it fits.

That truncation is in the wrong place. The winner was chosen for how good it is **deep**, and you
deploy it **shallow** — and those two rankings disagree. There is a second problem underneath it:

> ### Per-tree inference cost varies **7×–38×** across configurations of the same search space.
> *Measured across 23 tabular datasets; 23 of 23 cleared a pre-registered 2× threshold.*

At a 5 ms budget one configuration affords 400 trees and another affords 20. Ranking them at the
*same tree count* ranks them at *wildly different latencies* — so the comparison you actually care
about never happens.

## The fix

```
       budget: 5 ms/batch
              │
              ▼
  ┌───────────────────────┐
  │ 1. PROFILE            │   fit each candidate shallow, time it:
  │    latency ≈ p + q·k  │   q varies 7–38× — it cannot be predicted from
  └───────────┬───────────┘   hyperparameters, so it is measured
              ▼
  ┌───────────────────────┐   cfg A (cheap trees) → 400 trees affordable
  │ 2. BUDGET → CAPS      │   cfg B (deep trees)  →  20 trees affordable
  │    per configuration  │   floored onto real depths, never rounded up
  └───────────┬───────────┘
              ▼
  ┌───────────────────────┐   successive halving inside each cap;
  │ 3. SEARCH             │   cheap-per-tree configs climb higher, and
  │    compare at K_c     │   everything competes where it will run
  └───────────┬───────────┘
              ▼
  ┌───────────────────────┐   re-profile at deployment depth, shrink until
  │ 4. ENFORCE            │   the MEASUREMENT fits, minus a safety margin
  └───────────┬───────────┘
              ▼
     best_n_trees_ = 96, measured 3.97 ms  ✓
```

Step 4 is not decoration. Extrapolating a shallow-probe cost model to deep models under-predicts,
and an early build overshot its budget by ~15%. The value in `measured_latency_` is the
measurement that *decided* the depth — not a second draw that can disagree with the first.

## Install

```bash
pip install "isotune[xgboost]"      # or: isotune[lightgbm]
```

<details>
<summary><b>From source</b></summary>

```bash
git clone https://github.com/HeshamXOR/isotune && cd isotune
pip install -e ".[dev]"
pytest -q
python examples/smoke_end_to_end.py
```
</details>

## Does this apply to you?

| ✅ Use `isotune` when | ❌ Skip it when |
|---|---|
| Real-time scoring under an SLA — fraud, ad ranking, bidding, risk | Latency is irrelevant — use `RandomizedSearchCV`, it's simpler |
| Edge or on-device inference with a fixed hardware budget | Feature *computation* is the bottleneck, not tree traversal |
| High-QPS serving where p99 sets your machine count and your bill | Your model has no prefix scoring (not a boosted tree ensemble) |
| Batch jobs with a wall-clock window that cannot slip | You need a hard real-time *guarantee*, not a measured typical |

> **The clearest signal you need this:** you currently tune to convergence, then hand-lower
> `n_estimators` until it's fast enough. That manual step is exactly what this automates —
> *inside* the search rather than after it.

## Results

Same data, same search space, three budgets:

| Budget | Trees delivered | Measured | Test AUC |
|--------:|----------------:|---------:|---------:|
| 2 ms | 40 | 1.70 ms ✓ | 0.9632 |
| 5 ms | 96 | 3.97 ms ✓ | 0.9657 |
| 15 ms | 400 | 10.68 ms ✓ | 0.9672 |

Depth and accuracy both scale with the budget, and every delivered model meets it.

## API

| Object | Purpose |
|---|---|
| `IsoLatencySearchCV` | The search. sklearn-shaped: `fit`, `predict`, `predict_proba`. |
| `profile_booster(model, X)` | Measure `(p, q)` for a fitted XGBoost/LightGBM model. |
| `affordable_trees(p, q, budget)` | Deepest affordable depth. Rounds **down**; `0` means infeasible. |
| `budget_from_reference(p, q, k)` | A budget under which the *median* config affords exactly `k` trees. |
| `measure_latency(predict_at, prefixes)` | Fit `p + q·k` from your own timing callable. |

After `fit`: `best_n_trees_`, `measured_latency_`, `latency_budget_`, `latency_ok_`,
`predicted_latency_`, `profile_`, `excluded_`.

<details>
<summary><b>Why measure per-tree cost instead of predicting it from hyperparameters?</b></summary>

Because prediction doesn't work. Pooled Spearman correlation between `max_depth` and measured
per-tree cost is only **+0.588**, and the strongest predictor *changes by dataset* —
`min_child_weight` carries it on one (−0.830) and is nearly useless on another (−0.127). A library
that priced latency from hyperparameters alone would mis-allocate. So `isotune` measures `q`
directly, once per configuration, after it has been fitted to any depth. Prefix scoring makes that
cheap: one fitted model answers every depth below it.

</details>

<details>
<summary><b>Batch size matters more than you'd expect</b></summary>

Below roughly 512 rows, fixed per-call overhead dominates and swamps the per-tree term the
allocator needs. **Profile at the batch size you will actually serve** — use `batch_size=1` for
single-row online serving, not the 2048 default.

</details>

## What's validated, and what isn't

Being precise, because these are two different claims.

**Validated.** The *allocation rule* — give each configuration its own affordable depth, rather
than one global depth plus truncation — was pre-registered with thresholds fixed in advance and
confirmed on held-out data:

<div align="center">

**+0.0033 mean test AUC** over the truncate-afterwards baseline<br>
**11 of 11** held-out datasets · **Wilcoxon p = 0.0010**<br>
<sub>replicating +0.0027, 12 of 12, on a disjoint development set</sub>

</div>

The effect concentrates at tight budgets (+0.0034 and +0.0073 at the two tightest of four) and is
small at loose ones — which is what the mechanism predicts, since a loose budget is barely a
constraint.

**Not validated.** *This package* reimplements that rule for live models and has **not** been
through that benchmark. The confirmed numbers are evidence for the method, not a performance
promise for this code path.

**Scope.** Binary classification, XGBoost and LightGBM, CPU, measured on one machine at one batch
size. Latency rankings across hardware and batch sizes are untested. Multiclass and regression are
unimplemented rather than unsupported in principle.

## Related work

A budgeted-learning line for boosted trees already exists and should be cited alongside this —
[Greedy Miser](https://icml.cc/2012/papers/592.pdf) (Xu et al., ICML 2012), Speedboost,
[Gradient Regularized Budgeted Boosting](https://arxiv.org/abs/1901.04065),
[Optimally Pruning Decision Tree Ensembles With Feature Cost](https://arxiv.org/abs/1601.00955).

Those budget **feature-acquisition cost** and modify **training**. `isotune` budgets **traversal
latency** and constrains **selection**. Different quantity, different stage.

## License

MIT — see [LICENSE](LICENSE).
