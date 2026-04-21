import pytest
import numpy as np
from dml.models.irm import IRM
from dml.learners.lasso import LassoLearner

@pytest.fixture
def irm_data():
    np.random.seed(66)
    n = 500
    X = np.random.randn(n, 5)
    D = (np.random.randn(n) > 0).astype(float)
    Y = D * 1.0 + np.sin(X[:, 0]) + np.random.randn(n)
    theta_0 = 1.0
    return X, Y, D, theta_0

def test_irm_returns_results(irm_data):
    X, Y, D, _ = irm_data
    irm = IRM(learner=LassoLearner(), n_splits=5, random_state=66)
    irm.fit(Y, D, X)
    results = irm.predict()
    assert set(results.keys()) >= {'theta', 'var', 'ci_lower', 'ci_upper'}

def test_irm_theta_finite(irm_data):
    X, Y, D, _ = irm_data
    irm = IRM(learner=LassoLearner(), n_splits=5, random_state=66)
    irm.fit(Y, D, X)
    results = irm.predict()
    assert np.isfinite(results['theta'])

def test_irm_ci_contains_truth(irm_data):
    X, Y, D, theta_0 = irm_data
    irm = IRM(learner=LassoLearner(), n_splits=5, random_state=66)
    irm.fit(Y, D, X)
    results = irm.predict()
    assert results['ci_lower'] < theta_0 < results['ci_upper']

def test_irm_var_positive(irm_data):
    X, Y, D, _ = irm_data
    irm = IRM(learner=LassoLearner(), n_splits=5, random_state=66)
    irm.fit(Y, D, X)
    results = irm.predict()
    assert results['var'] > 0

def test_irm_theta_close_to_truth(irm_data):
    X, Y, D, theta_0 = irm_data
    irm = IRM(learner=LassoLearner(), n_splits=5, random_state=66)
    irm.fit(Y, D, X)
    results = irm.predict()
    assert abs(results['theta'] - theta_0) < 0.2