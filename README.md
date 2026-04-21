# DML from Scratch: A Controlled Comparison of Nuisance Learners

**Course:** ECON 622 — UBC

This project implements the Double Machine Learning (DML) framework from scratch, based on Chernozhukov et al. (2018). The goal is to study how learner choice, hyperparameter tuning, and DGP structure affect estimation quality in finite samples — using a controlled experimental setup that existing packages do not easily support.

---

## Motivation

Existing DML packages (e.g., DoubleML) are mature but hard to modify or verify at the component level. Building from scratch allows full control over each part of the pipeline, enabling systematic comparisons across learners under identical conditions.

---

## Framework Structure

The framework is organized into three layers:

**Learner Layer** — estimates nuisance functions g₀(X) = E[Y|X] and m₀(X) = E[D|X]:
- `LassoLearner` — parametric, sparse linear
- `RandomForestLearner` — tree-based, nonparametric
- `NeuralNetLearner` — deep learning
- `ElasticNetLearner` — appendix
- `CausalForestLearner` — appendix
- Tuned variants (`TunedLassoLearner`, `TunedRandomForestLearner`, `TunedNeuralNetLearner`)

**Cross-Fitting Layer** — generates out-of-sample residuals:
- `cross_fit` — standard K-fold cross-fitting
- `cross_fit_aggregated` — repeated cross-fitting averaged over multiple random splits
- `cross_fit_honest` — separate data for model selection and estimation

**Model Layer** — estimates the causal parameter θ:
- `PLR` — Partially Linear Regression using Robinson-style partialling-out score (Score 4.4, Chernozhukov et al. 2018)
- `IRM` — Interactive Regression Model using doubly robust ATE score (Section 5.1, Chernozhukov et al. 2018)

---

## Verification

Before running experiments, the framework is verified on three dimensions:

**Neyman Orthogonality** — JAX-based directional derivative check. Small perturbations are introduced to estimated nuisance functions, and derivatives along random directions are computed. Results close to zero confirm the score satisfies orthogonality:
- PLR: max |deriv| = 0.0156 ✅
- IRM: max |deriv| = 0.0045 ✅

**DoubleML Benchmark** — PLR estimates compared against the official DoubleML package on the same CCDDHNR2018 DGP. Difference < 0.01 ✅

**Semiparametric Efficiency Bound** — estimated variance compared against the theoretical lower bound for each learner.

---

## Experiments

All main experiments use the CCDDHNR2018 DGP (PLR, θ₀ = 0.5, dim_x = 20) unless otherwise noted.

### Experiment 1 — Estimator Validation
Compares three estimators: non-orthogonal ML, DML without sample splitting, and full DML with cross-fitting. Replicates the design of Chernozhukov et al. (2018) Figure 1 using real learners (Lasso, Neural Network). Validates that orthogonality and cross-fitting each contribute to estimation quality.

### Experiment 2 — Learner Comparison (Default Hyperparameters)
Compares Lasso, Random Forest, and Neural Network across sample sizes n ∈ {200, 500, 1000, 2000} with 500 Monte Carlo replications. Metrics: bias, RMSE, 95% CI coverage. ElasticNet results in appendix.

Key finding: all three learners achieve coverage near 95% at large n, but Random Forest shows coverage decline at n = 2000, consistent with slower nuisance convergence.

### Experiment 3 — Convergence Rate Analysis
Quantifies nuisance RMSE and θ̂ RMSE as a function of n on a log-log scale. Two sub-experiments:
- **(3a)** Nuisance RMSE vs n — checks whether learners satisfy the o(n^{-1/4}) rate condition
- **(3b)** θ̂ RMSE vs n — checks whether θ̂ achieves the parametric n^{-1/2} rate

Key finding: nuisance convergence is slow (especially for g₀), but θ̂ converges near n^{-1/2} for all learners, demonstrating Neyman orthogonality in practice.

### Experiment 4 — Tuned Learner Comparison
Repeats Experiment 2 with tuned hyperparameters:
- Lasso: LassoCV (automatic α selection)
- Random Forest: GridSearchCV over max_depth ∈ {4, 5, 6}
- Neural Network: architecture scaled with sample size, dropout regularization

Compares tuned vs. default performance to quantify the effect of hyperparameter tuning.

### Experiment 5 — DGP Robustness
Tests whether learner rankings hold across different DGPs:
- Sparse linear DGP — favorable for Lasso
- Piecewise nonlinear DGP — favorable for tree-based methods
- Dense nonlinear DGP — favorable for Neural Network

Uses tuned learners. Metrics: bias, RMSE, coverage at n ∈ {200, 500, 1000}.

---

## Repository Structure

```
dml-nuisance-comparison/
├── dml/
│   ├── learners/          # LassoLearner, RandomForestLearner, NeuralNetLearner, ...
│   ├── models/            # PLR, IRM
│   └── utils/             # cross_fitting, orthogonality, variance, efficiency_bound
├── experiments/
│   ├── exp1_estimator_comparison.py
│   ├── exp2_learner_comparison.py
│   ├── exp3_convergence_rate.py
│   ├── exp4_tuned_learner_comparison.py
│   └── exp5_dgp_robustness.py
├── tests/
│   ├── test_learners.py
│   ├── test_plr.py
│   ├── test_irm.py
│   ├── test_cross_fitting.py
│   └── test_causal_forest.py
├── results/               # output figures
└── notebooks/             # demo and experiment notebooks
```

---

## Installation

```bash
git clone https://github.com/yanhyuany/dml-nuisance-comparison.git
cd dml-nuisance-comparison
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Running Experiments

```python
from experiments.exp2_learner_comparison import run_experiment_2, plot_experiment_2

df = run_experiment_2(n_reps=500)
plot_experiment_2(df, save_path='results/exp2_learner_comparison.png')
```

---

## Reference

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
