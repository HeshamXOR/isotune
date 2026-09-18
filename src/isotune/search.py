"""`IsoLatencySearchCV` -- hyperparameter search whose deliverable meets a latency budget.

Drop-in shaped like `sklearn.model_selection.RandomizedSearchCV`, with one extra required
argument: `max_latency`, in seconds per batch. What comes back is the best configuration **at a
depth it can actually serve**, together with the measured latency of that exact model.

THE IDEA, in one paragraph. Standard practice tunes to convergence and truncates the winner
afterwards to fit the latency budget. That puts the truncation in the wrong place: the winner was
chosen for how good it is deep, not for how good it is at the depth it will be deployed at, and
those rank differently. `IsoLatencySearchCV` measures each configuration's per-tree cost, converts
the budget into a per-configuration tree cap, and runs successive halving inside those caps --- so
a configuration with cheap shallow trees is allowed to climb higher than an expensive one, and
they are compared where they will actually run.

WHAT IS AND IS NOT VALIDATED. The allocation rule was pre-registered and confirmed on 23 tabular
datasets via replay (+0.0033 mean test AUC over the truncate-afterwards baseline, 11/11 held-out
datasets, Wilcoxon p=0.0010). **This class is a reimplementation of that rule for live models and
has not itself been through that benchmark.** Treat the confirmed number as evidence for the
method, not as a performance promise for this code path.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, roc_auc_score)

from .budget import affordable_trees, budget_from_reference, latency_of
from .profile import DEFAULT_BATCH, profile_booster

__all__ = ["IsoLatencySearchCV"]

# Metrics applied directly to prefix predictions. All are oriented so that HIGHER IS BETTER,
# which is what the halving loop assumes; `neg_log_loss` carries the sign for that reason.
_PROBA_METRICS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
    "neg_log_loss": lambda y_true, y_score: -log_loss(y_true, y_score),
}
_LABEL_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": f1_score,
}


def _positive_scores(proba):
    """Positive-class column for binary problems, passed through otherwise."""
    proba = np.asarray(proba)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return proba


class IsoLatencySearchCV(BaseEstimator):
    """Latency-budgeted hyperparameter search for gradient-boosted trees.

    Parameters
    ----------
    estimator
        An unfitted XGBoost or LightGBM sklearn-API estimator. Must support prefix scoring
        (``iteration_range=`` or ``num_iteration=``), which both do.
    param_distributions
        As for ``RandomizedSearchCV``. Do **not** include the tree count -- it is the axis this
        class allocates, and passing it is an error.
    max_latency : float
        Budget in **seconds per batch of `batch_size` rows**, for the delivered model.
        If None, `reference_trees` must be given and the budget is derived so the median
        sampled configuration affords exactly that many trees.
    batch_size : int
        Rows per prediction batch, for both profiling and the budget's meaning. Profile at the
        size you will serve: below ~512 rows fixed overhead dominates and the per-tree term the
        allocator needs is hard to resolve. Use 1 for single-row online serving.
    n_candidates, max_trees, eta, probe_trees
        Search shape. `probe_trees` is the depth each configuration is fitted to for profiling.

    Attributes
    ----------
    best_estimator_, best_params_, best_n_trees_, best_score_
    measured_latency_ : float
        Latency of the delivered model, re-measured after selection rather than predicted.
    latency_budget_ : float
    profile_ : list of (p, q)
    excluded_ : int
        Configurations that could not afford a single tree and were dropped.
    """

    def __init__(self, estimator, param_distributions, max_latency=None, *, batch_size=DEFAULT_BATCH,
                 reference_trees=None, n_candidates=48, max_trees=512, eta=3, probe_trees=32, safety_margin=0.10,
                 scoring="roc_auc", validation_fraction=0.25, random_state=None, verbose=0):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.max_latency = max_latency
        self.batch_size = batch_size
        self.reference_trees = reference_trees
        self.n_candidates = n_candidates
        self.max_trees = max_trees
        self.eta = eta
        self.probe_trees = probe_trees
        self.safety_margin = safety_margin
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    # -- internals ---------------------------------------------------------------------------

    def _tree_param(self) -> str:
        return "n_estimators"

    def _fit_to(self, params, n_trees, X, y):
        est = clone(self.estimator)
        est.set_params(**{**params, self._tree_param(): int(n_trees)})
        est.fit(X, y)
        return est

    def _score_prefix(self, est, X, y, k):
        """Score a fitted model restricted to its first `k` trees.

        The metric is applied directly to the prefix predictions rather than through
        `sklearn.get_scorer`. Going through the scorer protocol means handing sklearn a wrapper
        object that must satisfy the full estimator contract (`__sklearn_tags__`, and whatever
        the next release adds), and a wrapper that fails that check raises from deep inside the
        scorer. This path depends only on `predict_proba`.

        Nothing is caught here. A search whose scores all silently become -inf still returns a
        model and still looks like it worked, which is the worst failure mode available -- it is
        how the first version of this class shipped a blind search that picked whichever
        configuration happened to be first.
        """
        view = _PrefixView(est, int(k))
        if callable(self.scoring):
            return float(self.scoring(y, _positive_scores(view.predict_proba(X))))
        if self.scoring in _PROBA_METRICS:
            return float(_PROBA_METRICS[self.scoring](y, _positive_scores(view.predict_proba(X))))
        if self.scoring in _LABEL_METRICS:
            return float(_LABEL_METRICS[self.scoring](y, view.predict(X)))
        raise ValueError(
            f"unsupported scoring {self.scoring!r}; use one of "
            f"{sorted(set(_PROBA_METRICS) | set(_LABEL_METRICS))} or pass a callable "
            "metric(y_true, y_score) where higher is better")

    def _log(self, msg):
        if self.verbose:
            print(f"[isotune] {msg}", flush=True)

    # -- public ------------------------------------------------------------------------------

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if self._tree_param() in getattr(self.param_distributions, "keys", lambda: [])():
            raise ValueError(
                f"do not search {self._tree_param()!r}: it is the axis IsoLatencySearchCV "
                "allocates from the latency budget"
            )
        rng = np.random.RandomState(self.random_state)
        Xtr, Xva, ytr, yva = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None)

        candidates = list(ParameterSampler(self.param_distributions, self.n_candidates,
                                           random_state=self.random_state))
        batch = Xva[rng.permutation(len(Xva))[:min(self.batch_size, len(Xva))]]

        # 1. Profile: fit each candidate shallow, measure its per-tree cost. This is the step
        #    that cannot be replaced by a formula over hyperparameters -- see profile.py.
        self._log(f"profiling {len(candidates)} candidates at {self.probe_trees} trees "
                  f"on a {len(batch)}-row batch")
        profile, probes = [], []
        for params in candidates:
            est = self._fit_to(params, self.probe_trees, Xtr, ytr)
            probes.append(est)
            profile.append(profile_booster(est, batch))
        p = np.array([a for a, _ in profile], dtype=float)
        q = np.array([b for _, b in profile], dtype=float)
        self.profile_ = list(zip(p.tolist(), q.tolist()))

        # 2. Budget -> per-configuration tree cap.
        if self.max_latency is None:
            if self.reference_trees is None:
                raise ValueError("give either max_latency or reference_trees")
            budget = budget_from_reference(p, q, self.reference_trees)
        else:
            budget = float(self.max_latency)
        self.latency_budget_ = budget
        caps = affordable_trees(p, q, budget, max_trees=self.max_trees)
        alive = [i for i in range(len(candidates)) if caps[i] >= 1]
        self.excluded_ = len(candidates) - len(alive)
        if not alive:
            raise RuntimeError(
                f"no candidate affords a single tree at {budget:.6g}s per {len(batch)} rows; "
                "raise max_latency, lower batch_size, or widen the space toward shallower trees")
        self._log(f"budget {budget:.6g}s -> caps {caps[alive].min()}-{caps[alive].max()} trees, "
                  f"{self.excluded_} excluded")

        # 3. Successive halving INSIDE the caps. A configuration is never taken past the depth it
        #    can serve; one whose cap is below the rung simply stays at its cap and competes there.
        rung, best = max(1, int(self.probe_trees)), None
        fitted = {i: probes[i] for i in alive}
        while alive:
            scores = {}
            for i in alive:
                depth = int(min(rung, caps[i]))
                if _n_fitted(fitted[i]) < depth:
                    fitted[i] = self._fit_to(candidates[i], depth, Xtr, ytr)
                scores[i] = (self._score_prefix(fitted[i], Xva, yva, depth), depth)
            for i, (s, d) in scores.items():
                if best is None or s > best[0]:
                    best = (s, i, d)
            self._log(f"rung {rung}: {len(alive)} alive, best so far {best[0]:.5f} "
                      f"(cfg {best[1]} @ {best[2]} trees)")
            if rung >= max(caps[alive].max(), 1):
                break
            keep = max(1, len(alive) // self.eta)
            alive = [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1][0])[:keep]]
            rung = int(rung * self.eta)

        # 4. Deliver, then ENFORCE the budget against a re-measurement rather than a prediction.
        #
        #    The profile in step 1 was fitted on a shallow probe and has to extrapolate to the
        #    selected depth, which under-predicts: measured latency came in ~15% over budget at
        #    tight budgets before this step existed. Extrapolating a cost model past its
        #    calibration range is a known way to get a confident wrong answer, so the final model
        #    is re-profiled over prefixes spanning the depth it will actually serve, and the depth
        #    is reduced until the measurement fits. Prefix scoring makes this nearly free -- one
        #    fitted model answers every depth below it, so no refit is needed to shrink.
        score, idx, depth = best
        self.best_params_ = dict(candidates[idx])
        est = self._fit_to(self.best_params_, depth, X, y)
        self.predicted_latency_ = latency_of(p[idx], q[idx], depth)

        # Aim at a fraction of the budget, not at the budget itself. Accepting a depth that is
        # marginally inside it means the next measurement -- on a slightly busier machine, or
        # just a different sample -- lands outside. The measurement that decides is the one
        # reported, so there is no second draw to disagree with the first.
        target = budget * (1.0 - float(self.safety_margin))
        measured = None
        for _ in range(6):
            grid = sorted({max(1, int(round(depth * f))) for f in (0.25, 0.5, 0.75, 1.0)})
            pm, qm = profile_booster(est, batch, prefixes=grid)
            measured = latency_of(pm, qm, depth)
            if measured <= target or depth <= 1:
                break
            shrunk = int(affordable_trees([pm], [qm], target, max_trees=depth)[0])
            new_depth = max(1, min(depth - 1, shrunk))
            self._log(f"re-measured {measured*1e3:.3f}ms > target {target*1e3:.3f}ms "
                      f"-> shrinking {depth} to {new_depth} trees")
            depth = new_depth

        self.best_score_ = float(score)
        self.best_n_trees_ = int(depth)
        self.best_estimator_ = est
        self.measured_latency_ = float(measured)
        self.latency_ok_ = bool(self.measured_latency_ <= budget)
        if not self.latency_ok_:
            self._log("WARNING: could not meet the budget even at the shallowest depth tried; "
                      "treat latency_ok_=False as a hard failure, not a rounding issue")
        self._log(f"selected cfg {idx} @ {depth} trees, score {score:.5f}, "
                  f"measured {self.measured_latency_*1e3:.3f}ms "
                  f"(budget {budget*1e3:.3f}ms, ok={self.latency_ok_})")
        return self

    def predict(self, X):
        return _PrefixView(self.best_estimator_, self.best_n_trees_).predict(X)

    def predict_proba(self, X):
        return _PrefixView(self.best_estimator_, self.best_n_trees_).predict_proba(X)


def _n_fitted(est) -> int:
    v = getattr(est, "n_estimators", None)
    return int(v) if v is not None else 0


class _PrefixView:
    """Restricts a fitted booster to its first `k` trees for scoring, without refitting.

    This is the reason prefix search is cheap: one fitted model answers every depth below it
    exactly, so a rung costs a prediction rather than a fit.
    """

    def __init__(self, est, k):
        self.est, self.k = est, int(k)
        self._estimator_type = getattr(est, "_estimator_type", "classifier")
        self.classes_ = getattr(est, "classes_", None)

    def __sklearn_tags__(self):
        # Delegate so the view can still be handed to sklearn utilities that expect a real
        # estimator. isotune's own scoring path does not rely on this.
        return self.est.__sklearn_tags__()

    def _call(self, name, X):
        fn = getattr(self.est, name)
        try:
            return fn(X, iteration_range=(0, self.k))
        except TypeError:
            return fn(X, num_iteration=self.k)

    def predict_proba(self, X):
        return self._call("predict_proba", X)

    def predict(self, X):
        return self._call("predict", X)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
