import pytest
import numpy as np
from dml.learners.lasso import LassoLearner
from dml.utils.cross_fitting import cross_fit, cross_fit_aggregated, cross_fit_honest


@pytest.fixture
def regression_data():
    np.random.seed(66)
    n = 300
    X = np.random.randn(n, 5)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n)
    return X, y


def test_cross_fit_output_shape(regression_data):
    X, y = regression_data
    y_pred = cross_fit(LassoLearner(), X, y, n_splits=5)
    assert y_pred.shape == y.shape


def test_cross_fit_all_filled(regression_data):
    X, y = regression_data
    y_pred = cross_fit(LassoLearner(), X, y, n_splits=5)
    assert np.all(np.isfinite(y_pred))


def test_cross_fit_reasonable_rmse(regression_data):
    X, y = regression_data
    y_pred = cross_fit(LassoLearner(), X, y, n_splits=5)
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    assert rmse < np.std(y)


def test_cross_fit_reproducible(regression_data):
    X, y = regression_data
    y1 = cross_fit(LassoLearner(), X, y, n_splits=5, random_state=66)
    y2 = cross_fit(LassoLearner(), X, y, n_splits=5, random_state=66)
    np.testing.assert_array_equal(y1, y2)


def test_cross_fit_different_seeds_differ(regression_data):
    X, y = regression_data
    y1 = cross_fit(LassoLearner(), X, y, n_splits=5, random_state=1)
    y2 = cross_fit(LassoLearner(), X, y, n_splits=5, random_state=2)
    assert not np.array_equal(y1, y2)


@pytest.mark.parametrize("n_splits", [2, 3, 5])
def test_cross_fit_various_splits(regression_data, n_splits):
    X, y = regression_data
    y_pred = cross_fit(LassoLearner(), X, y, n_splits=n_splits)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(y_pred))


def test_cross_fit_aggregated_output_shape(regression_data):
    X, y = regression_data
    y_pred = cross_fit_aggregated(LassoLearner(), X, y, n_splits=5, n_rep=3)
    assert y_pred.shape == y.shape


def test_cross_fit_aggregated_all_finite(regression_data):
    X, y = regression_data
    y_pred = cross_fit_aggregated(LassoLearner(), X, y, n_splits=5, n_rep=3)
    assert np.all(np.isfinite(y_pred))


def test_cross_fit_aggregated_smoother_than_single(regression_data):
    X, y = regression_data
    y_single = cross_fit(LassoLearner(), X, y, n_splits=5, random_state=66)
    y_agg = cross_fit_aggregated(LassoLearner(), X, y, n_splits=5, n_rep=10, random_state=66)
    assert np.var(y_agg) <= np.var(y_single) * 1.1


def test_cross_fit_aggregated_reproducible(regression_data):
    X, y = regression_data
    y1 = cross_fit_aggregated(LassoLearner(), X, y, n_splits=5, n_rep=3, random_state=66)
    y2 = cross_fit_aggregated(LassoLearner(), X, y, n_splits=5, n_rep=3, random_state=66)
    np.testing.assert_array_equal(y1, y2)


def test_cross_fit_honest_output_shape(regression_data):
    X, y = regression_data
    y_pred = cross_fit_honest(LassoLearner(), LassoLearner(), X, y, n_splits=5)
    assert y_pred.shape == y.shape


def test_cross_fit_honest_all_finite(regression_data):
    X, y = regression_data
    y_pred = cross_fit_honest(LassoLearner(), LassoLearner(), X, y, n_splits=5)
    assert np.all(np.isfinite(y_pred))


def test_cross_fit_honest_rest_is_out_of_sample(regression_data):
    X, y = regression_data
    n_select = int(len(y) * 0.2)
    y_pred = cross_fit_honest(LassoLearner(), LassoLearner(), X, y, n_splits=5)
    rest_preds = y_pred[n_select:]
    assert np.std(rest_preds) > 0