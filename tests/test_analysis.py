import numpy as np
import pandas as pd

from analysis import bootstrap_ci, mcnemar_test


class TestBootstrapCI:
    def test_perfect_data(self):
        data = np.ones(100)
        mean, lower, upper = bootstrap_ci(data)
        assert mean == 1.0
        assert lower == 1.0
        assert upper == 1.0

    def test_zero_data(self):
        data = np.zeros(100)
        mean, lower, upper = bootstrap_ci(data)
        assert mean == 0.0

    def test_ci_contains_mean(self):
        rng = np.random.RandomState(42)
        data = rng.binomial(1, 0.7, size=200)
        mean, lower, upper = bootstrap_ci(data)
        assert lower <= mean <= upper

    def test_ci_width_reasonable(self):
        rng = np.random.RandomState(42)
        data = rng.binomial(1, 0.5, size=1000)
        mean, lower, upper = bootstrap_ci(data)
        assert upper - lower < 0.1

    def test_deterministic_with_seed(self):
        data = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        r1 = bootstrap_ci(data, seed=42)
        r2 = bootstrap_ci(data, seed=42)
        assert r1 == r2


class TestMcNemarTest:
    def test_identical_predictions(self):
        a = np.array([1, 1, 0, 0, 1])
        result = mcnemar_test(a, a)
        assert result['p_value'] == 1.0
        assert result['a_right_b_wrong'] == 0
        assert result['a_wrong_b_right'] == 0

    def test_opposite_predictions(self):
        a = np.array([1, 1, 1, 0, 0])
        b = np.array([0, 0, 0, 1, 1])
        result = mcnemar_test(a, b)
        assert result['a_right_b_wrong'] == 3
        assert result['a_wrong_b_right'] == 2
        assert 0 <= result['p_value'] <= 1.0

    def test_significant_difference(self):
        a = np.concatenate([np.ones(100), np.zeros(10)])
        b = np.concatenate([np.zeros(100), np.ones(10)])
        result = mcnemar_test(a, b)
        assert result['p_value'] < 0.05

    def test_returns_expected_keys(self):
        a = np.array([1, 0, 1])
        b = np.array([0, 1, 1])
        result = mcnemar_test(a, b)
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'a_right_b_wrong' in result
        assert 'a_wrong_b_right' in result
