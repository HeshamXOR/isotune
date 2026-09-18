"""Tests for the budget arithmetic and the profiler.

The budget tests are exact and hand-computed -- this is the layer where an off-by-one hands back
a model that violates the budget it was selected under, so it is checked against arithmetic rather
than against itself.
"""

import numpy as np
import pytest

from isotune import affordable_trees, budget_from_reference, is_feasible, latency_of
from isotune.profile import measure_latency, time_callables


class TestBudget:
    def test_affordable_trees_is_hand_computable(self):
        # p = 1ms fixed, q = 0.1ms per tree, budget 5ms -> (5-1)/0.1 = 40 trees exactly.
        got = affordable_trees([1e-3], [1e-4], 5e-3)
        assert got.tolist() == [40]

    def test_rounding_is_always_down(self):
        # (5 - 1) / 0.3 = 13.33 -> 13, never 14. Rounding up would breach the budget.
        got = affordable_trees([1e-3], [3e-4], 5e-3)
        assert got.tolist() == [13]
        assert is_feasible(1e-3, 3e-4, 13, 5e-3)
        assert not is_feasible(1e-3, 3e-4, 14, 5e-3)

    def test_heterogeneity_produces_different_caps(self):
        # The premise of the whole library: same budget, very different affordable depths.
        p = [1e-3, 1e-3, 1e-3]
        q = [1e-5, 1e-4, 1e-3]          # 100x spread in per-tree cost
        caps = affordable_trees(p, q, 5e-3)
        assert caps.tolist() == [400, 40, 4]
        assert caps.max() / caps.min() == 100

    def test_unaffordable_configuration_returns_zero(self):
        # Fixed overhead alone exceeds the budget -> affords nothing, must be excluded.
        assert affordable_trees([10e-3], [1e-5], 5e-3).tolist() == [0]
        # And a per-tree cost so large that not even one tree fits.
        assert affordable_trees([4e-3], [2e-3], 5e-3).tolist() == [0]

    def test_grid_floors_onto_available_depths(self):
        caps = affordable_trees([1e-3], [1e-4], 5e-3, grid=[10, 25, 50, 100])
        assert caps.tolist() == [25]            # 40 affordable, 25 is the deepest grid point <= 40

    def test_max_trees_caps_the_result(self):
        assert affordable_trees([1e-3], [1e-9], 5e-3, max_trees=100).tolist() == [100]

    def test_budget_from_reference_gives_median_exactly_k(self):
        p = np.array([1e-3, 2e-3, 3e-3])
        q = np.array([1e-4, 2e-4, 3e-4])        # median config is index 1
        budget = budget_from_reference(p, q, 50)
        assert budget == pytest.approx(2e-3 + 2e-4 * 50)
        assert affordable_trees(p, q, budget)[1] == 50

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            affordable_trees([1e-3], [1e-4], 0)
        with pytest.raises(ValueError):
            affordable_trees([1e-3, 2e-3], [1e-4], 5e-3)
        with pytest.raises(ValueError):
            affordable_trees([np.nan], [1e-4], 5e-3)

    def test_latency_of_matches_the_affine_model(self):
        assert latency_of(1e-3, 1e-4, 40) == pytest.approx(5e-3)


class TestProfiler:
    def test_recovers_a_known_slope(self):
        # A synthetic "model" whose cost is genuinely affine in k: p=2ms, q=0.05ms per tree.
        def predict_at(k):
            t = time.perf_counter() + 2e-3 + 5e-5 * k
            while time.perf_counter() < t:
                pass

        import time
        p, q = measure_latency(predict_at, [10, 20, 40, 80], repeats=3, inner=1)
        assert q == pytest.approx(5e-5, rel=0.3)
        assert p == pytest.approx(2e-3, rel=0.5)

    def test_needs_two_prefixes_to_fit_a_slope(self):
        with pytest.raises(ValueError):
            measure_latency(lambda k: None, [10])

    def test_time_callables_returns_one_median_per_key(self):
        got = time_callables({"a": lambda: None, "b": lambda: None}, repeats=2, inner=1)
        assert set(got) == {"a", "b"}
        assert all(v >= 0 for v in got.values())


import time  # noqa: E402  (used inside TestProfiler.test_recovers_a_known_slope)
