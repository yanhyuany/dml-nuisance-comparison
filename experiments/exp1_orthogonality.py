import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from dml.learners.lasso import LassoLearner
from dml.models.plr import PLR
from dml.utils.cross_fitting import cross_fit

def generate_data(n_obs: int = 500, n_vars: int = 20,
                  alpha: float = 0.5, random_state: int = None):
    """
    Use the official CCDDHNR2018 DGP from Chernozhukov et al. (2018).
    Nonlinear nuisance functions, so OLS is misspecified.
    alpha is the true theta_0.
    """
    np.random.seed(random_state)
    X, Y, D = make_plr_CCDDHNR2018(
        alpha=alpha,
        n_obs=n_obs,
        dim_x=n_vars,
        return_type='array'
    )
    return X, Y, D

def estimate_nonorthogonal(X, Y, D, learner):
    """
    Non-orthogonal ML: estimate g(X) with ML, then regress Y - g_hat on D.
    No partialling out, no cross-fitting.
    """
    # fit g(X) = E[Y|X] on full data
    g_hat = learner.fit(X, Y).predict(X)
    Y_res = Y - g_hat

    # regress Y_res on D (no partialling out of D)
    theta = np.sum(D * Y_res) / np.sum(D * D)
    
    # variance estimate
    psi = D * (Y_res - D * theta)
    J = np.mean(D ** 2)
    var = np.mean(psi ** 2) / (J ** 2) / len(Y)
    se = np.sqrt(var)
    
    return theta, se


def estimate_dml_no_split(X, Y, D, learner):
    """
    DML without sample splitting: partialling out but no cross-fitting.
    Fit and predict on the same data.
    """
    # partial out Y
    Y_hat = learner.fit(X, Y).predict(X)
    Y_tilde = Y - Y_hat

    # partial out D
    D_hat = learner.fit(X, D).predict(X)
    D_tilde = D - D_hat

    # PLR score
    theta = (D_tilde @ Y_tilde) / (D_tilde @ D_tilde)

    # variance
    psi = D_tilde * (Y_tilde - D_tilde * theta)
    J = np.mean(D_tilde ** 2)
    var = np.mean(psi ** 2) / (J ** 2) / len(Y)
    se = np.sqrt(var)

    return theta, se


def estimate_dml_crossfit(X, Y, D, learner):
    """
    Full DML with cross-fitting: use our PLR implementation.
    """
    from dml.learners.lasso import LassoLearner
    plr = PLR(learner=learner, n_splits=5, random_state=66)
    plr.fit(Y, D, X)
    results = plr.predict()
    theta = results['theta']
    se = np.sqrt(results['var'])
    return theta, se

def run_experiment_1(n_obs: int = 500, n_vars: int = 20,
                     alpha: float = 0.5, n_reps: int = 500) -> dict:
    """
    Experiment 1: Compare three estimators using CCDDHNR2018 DGP.
    - Non-orthogonal ML
    - DML without sample splitting
    - DML with cross-fitting
    Returns standardized estimates (theta - theta_0) / se for each.
    """
    results = {
        "nonorth": [],
        "dml_nosplit": [],
        "dml_crossfit": []
    }

    for rep in range(n_reps):
        if rep % 50 == 0:
            print(f"Replication {rep}/{n_reps}...")

        X, Y, D = generate_data(n_obs, n_vars, alpha, random_state=rep)
        learner = LassoLearner()

        # non-orthogonal
        theta, se = estimate_nonorthogonal(X, Y, D, LassoLearner())
        results["nonorth"].append((theta - alpha) / se)

        # DML no split
        theta, se = estimate_dml_no_split(X, Y, D, LassoLearner())
        results["dml_nosplit"].append((theta - alpha) / se)

        # DML with cross-fitting
        theta, se = estimate_dml_crossfit(X, Y, D, LassoLearner())
        results["dml_crossfit"].append((theta - alpha) / se)

    return {k: np.array(v) for k, v in results.items()}

def plot_experiment_1(results: dict, save_path: str = None):
    """
    Plot standardized distribution of three estimators.
    Expected: only DML with cross-fitting matches N(0,1).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = {
        "nonorth": "Non-orthogonal ML",
        "dml_nosplit": "DML (no sample splitting)",
        "dml_crossfit": "DML (with cross-fitting)"
    }
    colors = ["coral", "orange", "steelblue"]
    xx = np.linspace(-6, 6, 200)
    yy = stats.norm.pdf(xx)

    for ax, (key, label), color in zip(axes, labels.items(), colors):
        ax.hist(results[key], bins=40, density=True,
                color=color, alpha=0.6, label=label)
        ax.plot(xx, yy, 'k-', linewidth=1.5, label='N(0,1)')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim([-6, 6])
        ax.set_xlabel(r'$(\hat{\theta} - \theta_0) / \hat{\sigma}$')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()