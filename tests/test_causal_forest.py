import pytest
import numpy as np
from dml.learners.causal_forest import CausalForestLearner
 
 
# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
 
@pytest.fixture
def regression_data():
    np.random.seed(66)
    n = 300
    X = np.random.randn(n, 5)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n)
    return X, y
 
 
# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------
 
def test_causal_forest_fit_returns_self(regression_data):
    X, y = regression_data
    learner = CausalForestLearner()
    result = learner.fit(X, y)
    assert result is learner
 
 
def test_causal_forest_predict_shape(regression_data):
    X, y = regression_data
    learner = CausalForestLearner()
    learner.fit(X, y)
    preds = learner.predict(X)
    assert preds.shape == (len(y),)
 
 
def test_causal_forest_predict_1d(regression_data):
    """Output must be 1-D (flatten() applied)."""
    X, y = regression_data
    learner = CausalForestLearner()
    learner.fit(X, y)
    preds = learner.predict(X)
    assert preds.ndim == 1
 
 
def test_causal_forest_predict_finite(regression_data):
    X, y = regression_data
    learner = CausalForestLearner()
    learner.fit(X, y)
    preds = learner.predict(X)
    assert np.all(np.isfinite(preds))
 
 
def test_causal_forest_predict_before_fit_raises():
    learner = CausalForestLearner()
    X = np.random.randn(10, 3)
    with pytest.raises(Exception):
        learner.predict(X)
 
 
# ---------------------------------------------------------------------------
# fit_predict (BaseNuisanceLearner interface)
# ---------------------------------------------------------------------------
 
def test_causal_forest_fit_predict_shape(regression_data):
    X, y = regression_data
    n = len(y)
    X_train, y_train = X[: n // 2], y[: n // 2]
    X_test = X[n // 2 :]
    learner = CausalForestLearner()
    preds = learner.fit_predict(X_train, y_train, X_test)
    assert preds.shape == (len(X_test),)
 
 
def test_causal_forest_fit_predict_finite(regression_data):
    X, y = regression_data
    n = len(y)
    X_train, y_train = X[: n // 2], y[: n // 2]
    X_test = X[n // 2 :]
    learner = CausalForestLearner()
    preds = learner.fit_predict(X_train, y_train, X_test)
    assert np.all(np.isfinite(preds))
 
 
# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
 
def test_causal_forest_reasonable_rmse(regression_data):
    """In-sample RMSE should beat predicting the mean."""
    X, y = regression_data
    learner = CausalForestLearner()
    learner.fit(X, y)
    preds = learner.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    assert rmse < np.std(y)
 
 
# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
 
def test_causal_forest_reproducible(regression_data):
    X, y = regression_data
    l1 = CausalForestLearner(random_state=42)
    l2 = CausalForestLearner(random_state=42)
    l1.fit(X, y)
    l2.fit(X, y)
    np.testing.assert_array_equal(l1.predict(X), l2.predict(X))
 
 
def test_causal_forest_different_seeds_differ(regression_data):
    X, y = regression_data
    l1 = CausalForestLearner(random_state=1)
    l2 = CausalForestLearner(random_state=2)
    l1.fit(X, y)
    l2.fit(X, y)
    assert not np.array_equal(l1.predict(X), l2.predict(X))
 
 
# ---------------------------------------------------------------------------
# Constructor parameters
# ---------------------------------------------------------------------------
 
@pytest.mark.parametrize("n_estimators", [50, 100, 200])
def test_causal_forest_n_estimators(regression_data, n_estimators):
    X, y = regression_data
    learner = CausalForestLearner(n_estimators=n_estimators)
    learner.fit(X, y)
    preds = learner.predict(X)
    assert preds.shape == (len(y),)