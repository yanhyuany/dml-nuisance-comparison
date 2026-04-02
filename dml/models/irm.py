import numpy as np
from ..utils.cross_fitting import cross_fit
from ..utils.variance import compute_variance, confidence_interval

class IRM:
    """
    Interactive Regression Model via Double/Debiased ML.
    
    Estimates ATE using doubly robust score with cross-fitting.
    D must be binary (0 or 1).
    """

    def __init__(self, learner, n_splits: int = 5, random_state: int = 66,
                 trim: float = 0.01):
        self.learner = learner
        self.n_splits = n_splits
        self.random_state = random_state
        self.trim = trim

        self.theta_ = None
        self.var_ = None
        self.ci_ = None

    def fit(self, Y: np.ndarray, D: np.ndarray, X: np.ndarray):
        n = len(Y)

        # step 1: cross-fit Y on X, get l(X) = E[Y|X]
        l_hat = cross_fit(self.learner, X, Y,
                          n_splits=self.n_splits,
                          random_state=self.random_state)

        # step 2: cross-fit D on X, get m(X) = P(D=1|X)
        m_hat = cross_fit(self.learner, X, D,
                          n_splits=self.n_splits,
                          random_state=self.random_state)

        # step 3: trim propensity score to avoid division by near-zero
        m_hat = np.clip(m_hat, self.trim, 1 - self.trim)

        # step 4: compute doubly robust score and estimate theta
        psi_b = (Y - l_hat) * (D - m_hat) / (m_hat * (1 - m_hat))
        psi_a = -np.ones(n)
        self.theta_ = -np.mean(psi_b) / np.mean(psi_a)

        # step 5: variance and CI
        psi = psi_b + psi_a * self.theta_
        J = np.mean(psi_a)
        self.var_ = compute_variance(psi, J, n)
        self.ci_ = confidence_interval(self.theta_, self.var_)

        return self
    
    def predict(self):
        if self.theta_ is None:
            raise ValueError("IRM model has not been fit yet. Call fit() first.")
        return {
            "theta": self.theta_,
            "var": self.var_,
            "ci_lower": self.ci_[0],
            "ci_upper": self.ci_[1]
        }