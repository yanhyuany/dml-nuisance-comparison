import pytest
import numpy as np
from dml.learners.base import BaseNuisanceLearner
from dml.learners.lasso import LassoLearner
from dml.learners.elastic_net import ElasticNetLearner
from dml.learners.random_forest import RandomForestLearner
from dml.learners.causal_forest import CausalForestLearner

@pytest.fixture
def sample_data():
    """Simple synthetic dataset for testing learners."""
    np.random.seed(66)
    n, p = 100, 5
    X = np.random.randn(n, p)
    y = X @ np.ones(p) + np.random.randn(n)
    return X, y

@pytest.mark.parametrize("learner", [
    LassoLearner(),
    ElasticNetLearner(),
    RandomForestLearner(),
    CausalForestLearner(),
])
def test_learner_is_base_instance(learner):
    """All learners must be instances of BaseNuisanceLearner."""
    assert isinstance(learner, BaseNuisanceLearner)


@pytest.mark.parametrize("learner", [
    LassoLearner(),
    ElasticNetLearner(),
    RandomForestLearner(),
    CausalForestLearner(),
])
def test_learner_has_fit_predict(learner):
    """All learners must have fit() and predict() methods."""
    assert hasattr(learner, "fit")
    assert hasattr(learner, "predict")
    assert hasattr(learner, "fit_predict")

@pytest.mark.parametrize("learner", [
    LassoLearner(),
    ElasticNetLearner(),
    RandomForestLearner(),
    CausalForestLearner(),
])
def test_predict_output_shape(learner, sample_data):
    """predict() must return a 1D array of shape (n,)."""
    X, y = sample_data
    learner.fit(X, y)
    y_pred = learner.predict(X)
    assert y_pred.shape == (len(y),)

@pytest.mark.parametrize("learner", [
    LassoLearner(),
    ElasticNetLearner(),
    RandomForestLearner(),
    CausalForestLearner(),
])
def test_fit_predict_reduces_error(learner, sample_data):
    """Fitted learner should predict better than the mean baseline."""
    X, y = sample_data
    learner.fit(X, y)
    y_pred = learner.predict(X)
    
    mse_model = np.mean((y - y_pred) ** 2)
    mse_baseline = np.mean((y - np.mean(y)) ** 2)
    
    assert mse_model < mse_baseline


def test_fit_predict_method(sample_data):
    """fit_predict() should return same shape as predict()."""
    X, y = sample_data
    n = len(y)
    X_train, X_test = X[:80], X[80:]
    y_train = y[:80]
    
    learner = LassoLearner()
    result = learner.fit_predict(X_train, y_train, X_test)
    assert result.shape == (20,)