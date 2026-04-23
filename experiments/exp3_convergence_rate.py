import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from dml.learners.lasso import LassoLearner
from dml.learners.random_forest import RandomForestLearner
from dml.learners.neural_net import NeuralNetLearner
from dml.learners.elastic_net import ElasticNetLearner
from dml.utils.cross_fitting import cross_fit
from dml.models.plr import PLR

LEARNERS = {
    "Lasso": LassoLearner,
    "RandomForest": RandomForestLearner,
    "NeuralNet": NeuralNetLearner,
}

LEARNERS_APPENDIX = {
    "ElasticNet": ElasticNetLearner,
}

N_VALUES = [200, 500, 1000, 2000, 5000]
ALPHA = 0.5
N_REPS = 100


def g0_structural(X):
    return np.exp(X[:, 0]) / (1 + np.exp(X[:, 0])) + 0.25 * X[:, 2]


def m0_true(X):
    return X[:, 0] + 0.25 * np.exp(X[:, 2]) / (1 + np.exp(X[:, 2]))


def l0_true(X, alpha: float = ALPHA):
    return g0_structural(X) + alpha * m0_true(X)

def generate_data(n_obs: int, alpha: float = ALPHA, random_state: int = None):
    np.random.seed(random_state)
    X, Y, D = make_plr_CCDDHNR2018(
        alpha=alpha, n_obs=n_obs, dim_x=20, return_type='array'
    )
    return X, Y, D


def run_nuisance_rmse(learner_class, n_obs: int, random_state: int) -> dict:
    X, Y, D = generate_data(n_obs, random_state=random_state)
    l_hat = cross_fit(learner_class(), X, Y, n_splits=5, random_state=random_state)
    m_hat = cross_fit(learner_class(), X, D, n_splits=5, random_state=random_state)

    rmse_l = np.sqrt(np.mean((l_hat - l0_true(X, ALPHA)) ** 2))
    rmse_m = np.sqrt(np.mean((m_hat - m0_true(X)) ** 2))
    return {"rmse_l": rmse_l, "rmse_m": rmse_m}


def run_exp3a(n_reps: int = N_REPS, learners: dict = None) -> pd.DataFrame:
    if learners is None:
        learners = LEARNERS

    records = []
    for learner_name, learner_class in learners.items():
        for n_obs in N_VALUES:
            print(f"[{learner_name}] n={n_obs}...")
            rmse_l_list, rmse_m_list = [], []
            for rep in range(n_reps):
                try:
                    res = run_nuisance_rmse(learner_class, n_obs, random_state=rep)
                    rmse_l_list.append(res["rmse_l"])
                    rmse_m_list.append(res["rmse_m"])
                except Exception:
                    continue

            if len(rmse_l_list) == 0:
                records.append({
                    "learner": learner_name,
                    "n_obs": n_obs,
                    "rmse_l": np.nan,
                    "rmse_m": np.nan,
                    "n_valid": 0
                })
                print("  all replications failed.")
                continue

            records.append({
                "learner": learner_name,
                "n_obs": n_obs,
                "rmse_l": np.mean(rmse_l_list),
                "rmse_m": np.mean(rmse_m_list),
                "n_valid": len(rmse_l_list)
            })
    return pd.DataFrame(records)

def run_theta_rmse(learner_class, n_obs: int, random_state: int) -> float:
    X, Y, D = generate_data(n_obs, random_state=random_state)
    plr = PLR(learner=learner_class(), n_splits=5, random_state=random_state)
    plr.fit(Y, D, X)
    results = plr.predict()
    return (results["theta"] - ALPHA) ** 2


def run_exp3b(n_reps: int = N_REPS, learners: dict = None) -> pd.DataFrame:
    if learners is None:
        learners = LEARNERS

    records = []
    for learner_name, learner_class in learners.items():
        for n_obs in N_VALUES:
            print(f"[{learner_name}] n={n_obs}...")
            sq_errors = []
            for rep in range(n_reps):
                try:
                    se = run_theta_rmse(learner_class, n_obs, random_state=rep)
                    sq_errors.append(se)
                except Exception:
                    continue

            if len(sq_errors) == 0:
                records.append({
                    "learner": learner_name,
                    "n_obs": n_obs,
                    "rmse_theta": np.nan,
                    "n_valid": 0
                })
                print("  all replications failed.")
                continue

            records.append({
                "learner": learner_name,
                "n_obs": n_obs,
                "rmse_theta": np.sqrt(np.mean(sq_errors)),
                "n_valid": len(sq_errors)
            })
    return pd.DataFrame(records)


def estimate_slope(n_values, rmse_values) -> float:
    n_values = np.array(n_values, dtype=float)
    rmse_values = np.array(rmse_values, dtype=float)
    mask = np.isfinite(rmse_values) & (rmse_values > 0)
    log_n = np.log(n_values[mask])
    log_rmse = np.log(rmse_values[mask])
    slope, _ = np.polyfit(log_n, log_rmse, 1)
    return slope


def plot_exp3a(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    learner_colors = {
        "Lasso": "steelblue",
        "RandomForest": "green",
        "NeuralNet": "purple",
        "ElasticNet": "coral",
    }

    for nuisance, ax, title in zip(
        ["rmse_l", "rmse_m"],
        axes,
        ["Nuisance RMSE: l₀(X) = E[Y|X]", "Nuisance RMSE: m₀(X) = E[D|X]"]
    ):
        for learner_name in df["learner"].unique():
            subset = df[df["learner"] == learner_name]
            color = learner_colors.get(learner_name, "gray")
            rmse_vals = subset[nuisance].values
            slope = estimate_slope(N_VALUES[:len(rmse_vals)], rmse_vals)
            ax.loglog(subset["n_obs"], rmse_vals,
                      marker='o', color=color,
                      label=f"{learner_name} (slope={slope:.2f})")

        ref_n = np.array(N_VALUES, dtype=float)
        first_vals = df[df["learner"] == list(df["learner"].unique())[0]]
        c = first_vals[nuisance].values[0] * N_VALUES[0] ** 0.25
        ax.loglog(ref_n, c * ref_n ** (-0.25), 'k--',
                  linewidth=1.5, label='n^{-1/4} reference')

        ax.set_xlabel('Sample size (n)')
        ax.set_ylabel('RMSE')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle('Exp 3a: Nuisance Convergence Rate', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_exp3b(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 5))

    learner_colors = {
        "Lasso": "steelblue",
        "RandomForest": "green",
        "NeuralNet": "purple",
        "ElasticNet": "coral",
    }

    for learner_name in df["learner"].unique():
        subset = df[df["learner"] == learner_name]
        color = learner_colors.get(learner_name, "gray")
        rmse_vals = subset["rmse_theta"].values
        slope = estimate_slope(N_VALUES[:len(rmse_vals)], rmse_vals)
        ax.loglog(subset["n_obs"], rmse_vals,
                  marker='o', color=color,
                  label=f"{learner_name} (slope={slope:.2f})")

    ref_n = np.array(N_VALUES, dtype=float)
    first = df[df["learner"] == list(df["learner"].unique())[0]]
    c = first["rmse_theta"].values[0] * N_VALUES[0] ** 0.5
    ax.loglog(ref_n, c * ref_n ** (-0.5), 'k--',
              linewidth=1.5, label='n^{-1/2} reference')

    ax.set_xlabel('Sample size (n)')
    ax.set_ylabel('RMSE(θ̂)')
    ax.set_title('Exp 3b: θ̂ Convergence Rate')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()