import pytest
import numpy as np
from dml.models.plr import PLR
from dml.learners.lasso import LassoLearner

@pytest.fixture
def plr_data():
    np.random.seed(66)
    n, p = 500, 20
    X = np.random.randn(n, p)
    theta_0 = 1.0
    D = X @ np.random.randn(p) + np.random.randn(n)
    Y = theta_0 * D + X @ np.random.randn(p) + np.random.randn(n)
    return X, Y, D, theta_0

def test_plr_returns_results(plr_data):
    X, Y, D, _ = plr_data
    plr = PLR(learner=LassoLearner(), n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    assert set(results.keys()) >= {'theta', 'var', 'ci_lower', 'ci_upper'}

def test_plr_theta_finite(plr_data):
    X, Y, D, _ = plr_data
    plr = PLR(learner=LassoLearner(), n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    assert np.isfinite(results['theta'])

def test_plr_ci_contains_truth(plr_data):
    X, Y, D, theta_0 = plr_data
    plr = PLR(learner=LassoLearner(), n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    assert results['ci_lower'] < theta_0 < results['ci_upper']

def test_plr_var_positive(plr_data):
    X, Y, D, _ = plr_data
    plr = PLR(learner=LassoLearner(), n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    assert results['var'] > 0

def test_plr_theta_close_to_truth(plr_data):
    X, Y, D, theta_0 = plr_data
    plr = PLR(learner=LassoLearner(), n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    assert abs(results['theta'] - theta_0) < 0.2