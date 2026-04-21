import numpy as np


def compute_variance(psi: np.ndarray, J: float, n: int) -> float:
    return (1 / J**2) * np.mean(psi**2) / n


def confidence_interval(theta: float, var: float, alpha: float = 0.05):
    z = 1.96
    se = np.sqrt(var)
    return (theta - z * se, theta + z * se)