# DML from Scratch: A Controlled Comparison of Nuisance Learners

**Course:** ECON 622 — UBC | **Instructor:** Paul Schrimpf

This project implements the Double Machine Learning (DML) framework from scratch, based on Chernozhukov et al. (2018). The goal is to study how learner choice, hyperparameter tuning, and DGP structure affect estimation quality in finite samples — using a controlled experimental setup that existing packages do not easily support.

---

## Motivation

Existing DML packages (e.g., DoubleML) are mature but hard to modify or verify at the component level. Specifically, they do not allow isolated testing of orthogonality, swapping cross-fitting strategies, or plugging in custom learners while holding all other conditions fixed. Building from scratch allows full control over each part of the pipeline, enabling systematic comparisons across learners under identical conditions.

---

## Framework Structure

The framework is organized into three layers.

### Learner Layer

Estimates nuisance functions g₀(X) = E[Y|X] and m₀(X) = E[D|X]. All learners inherit from `BaseNuisanceLearner` (abstract base class), which enforces a common `fit()` / `predict()` / `fit_predict()` interface.

| Learner | Class | Default Parameters |
|---------|-------|--------------------|
| Lasso | `LassoLearner` | alpha=0.1 |
| Lasso (tuned) | `TunedLassoLearner` | LassoCV, cv=5 |
| Random Forest | `RandomForestLearner` | n_estimators=100, random_state=42 |
| Random Forest (tuned) | `TunedRandomForestLearner` | GridSearchCV: max_depth∈{4,5,6}, max_features∈{0.3,0.5,0.7}, n_jobs=-1 |
| Neural Network | `NeuralNetLearner` | hidden=[64,32], lr=1e-3, Adam, MSELoss, batch_size=64, early stopping (patience=10) |
| Neural Network (tuned) | `TunedNeuralNetLearner` | Farrell et al. (2021) architecture: depth=floor(log n), width=floor(n^{1/(2+p/n)}), dropout=0.1 |
| ElasticNet | `ElasticNetLearner` | alpha=0.1, l1_ratio=0.5 (appendix) |
| CausalForest | `CausalForestLearner` | n_estimators=100, econml RegressionForest (appendix) |

All neural network learners use early stopping with a validation split (val_frac=0.1), random shuffle before splitting, and mini-batch training with batches randomly reshuffled each epoch. The TunedRandomForestLearner uses `n_jobs=-1` to parallelize the grid search across all available CPU cores.

### Cross-Fitting Layer

Generates out-of-sample residuals to satisfy the sample-splitting requirement of DML. All functions use `n_splits=5` and `random_state=66` by default.

- **`cross_fit`**: Standard K-fold cross-fitting. Each observation's prediction is made by a model trained on the other K-1 folds, ensuring every prediction is out-of-sample.
- **`cross_fit_aggregated`**: Repeats cross-fitting `n_rep` times with different random fold assignments and averages predictions to reduce fold-assignment variance.
- **`cross_fit_honest`**: Reserves 20% of data as a selection set for model selection, then runs standard cross-fitting on the remaining 80%. Separates model selection from estimation. The selection set receives the mean of the training predictions as a placeholder.

### Model Layer

Estimates the causal parameter θ using cross-fitted nuisance predictions.

**PLR (Partially Linear Regression)** — Robinson partialling-out score (Score 4.4, Chernozhukov et al. 2018):

```
Y = θ·D + g₀(X) + ε
ψ(W; θ, η) = D̃·(Ỹ - θ·D̃)
θ̂ = (D̃ᵀỸ) / (D̃ᵀD̃)
```

where Ỹ = Y - ĝ(X) and D̃ = D - m̂(X) are cross-fitted residuals. θ̂ is the OLS solution from regressing Ỹ on D̃.

**IRM (Interactive Regression Model)** — Doubly robust ATE score (Section 5.1, Chernozhukov et al. 2018):

```
ψ(W; θ, η) = g(1,X) - g(0,X) + D(Y - g(1,X))/m(X) - (1-D)(Y - g(0,X))/(1-m(X)) - θ
θ̂ = -E[ψ_b] / E[ψ_a],  where ψ_a = -1
```

Three nuisance functions are cross-fitted separately using the same fold structure: g(0,X) = E[Y|D=0,X], g(1,X) = E[Y|D=1,X], m(X) = P(D=1|X). Propensity scores are clipped to [0.01, 0.99] to prevent division by zero. The clipping bounds are chosen to be conservative — wide enough to avoid numerical instability while not discarding too much variation in the propensity score.

Variance is estimated via the influence function: V̂ = (1/J²) · E[ψ²] / n, where J = E[D̃²] for PLR and J = E[ψ_a] = -1 for IRM. 95% confidence intervals use the normal approximation with z = 1.96.

---

## Verification

Before running experiments, the framework is verified on three dimensions.

### Neyman Orthogonality (JAX-based)

Small perturbations are introduced to estimated nuisance functions along random directions h ~ N(0,1) (mean-centered), and directional derivatives are computed using JAX automatic differentiation:

```
φ'(0) = d/dt E[ψ(W; θ₀, η̂ + t·h)] |_{t=0}
```

This is repeated over 10 independent random directions per nuisance function. Results close to zero provide numerical support for orthogonality. Note: this is a numerical check using estimated nuisance functions, not a formal proof — residual estimation error means derivatives will not be exactly zero in finite samples.

- PLR: max |deriv| = 0.0156 ✅ (checked on l(X) and m(X) directions)
- IRM: max |deriv| = 0.0045 ✅ (checked on g₀, g₁, m(X) directions)

### DoubleML Benchmark

PLR estimates compared against the official DoubleML package on the CCDDHNR2018 DGP (n=500, dim_x=20, alpha=0.5, LassoLearner). Difference in θ̂ < 0.01 ✅

IRM benchmark uses a regressor with output clipped to [0.01, 0.99] for propensity score estimation, while the official DoubleML package uses a separate classifier. Results may therefore not be fully attributed to implementation correctness alone.

### Semiparametric Efficiency Bound

Estimated variance compared against the theoretical lower bound V* = σ²/J²/n. Also verified using JAX-based automatic differentiation to compute the Jacobian J independently.

---

## Experiments

All experiments use `random_state=66` as the default seed. Monte Carlo replications use `random_state=rep` (0-indexed) for both data generation and PLR/IRM fitting. Main experiments use the CCDDHNR2018 DGP (PLR, θ₀=0.5, dim_x=20) unless otherwise noted. All PLR and IRM fits use `n_splits=5`. Failed replications are skipped via `try/except` and counted via `n_valid`.

### Experiment 1 — Estimator Validation

Compares three estimators on the same DGP (n=500, 500 Monte Carlo replications):

- **Non-orthogonal ML**: in-sample prediction, no partialling-out
- **DML without sample splitting**: orthogonalization but in-sample nuisance estimation
- **DML with cross-fitting**: full implementation

Each estimator produces a standardized statistic (θ̂ - θ₀)/σ̂, plotted against the N(0,1) reference. Lasso bias in the non-orthogonal case comes from two sources: λ regularization and model misspecification (linear model cannot fit nonlinear g₀). NN improves most dramatically from no-split to cross-fitting, consistent with higher overfitting risk.

### Experiment 2 — Learner Comparison (Default Hyperparameters)

Compares Lasso, Random Forest, and Neural Network across n ∈ {200, 500, 1000, 2000} with 500 Monte Carlo replications. Metrics: bias, RMSE, 95% CI coverage. ElasticNet and CausalForest results in appendix.

### Experiment 3 — Convergence Rate Analysis

Quantifies nuisance RMSE and θ̂ RMSE as a function of n ∈ {200, 500, 1000, 2000, 5000} on a log-log scale (100 replications). Uses the known true nuisance functions of the CCDDHNR2018 DGP:

```
g₀(X) = exp(X₁)/(1+exp(X₁)) + 0.25·X₃
m₀(X) = X₁ + 0.25·exp(X₃)/(1+exp(X₃))
```

Convergence slope estimated via log-log OLS. Theory predicts nuisance slope < -0.25 (o(n^{-1/4})) and θ̂ slope ≈ -0.5 (n^{-1/2}). Observed θ̂ slopes range from -0.40 to -0.47 across learners, consistent with theory. Nuisance slopes for g₀ are near zero, reflecting that the nonlinear DGP and the finite n range place learners outside the asymptotic regime for this nuisance function.

### Experiment 4 — Tuned Learner Comparison

Repeats Experiment 2 with tuned hyperparameters. Compares baseline vs tuned side-by-side (2×3 subplot grid). Hyperparameter tuning yields minimal improvement on RMSE but provides some improvement on CI coverage, particularly for Random Forest at large n.

### Experiment 5 — DGP Robustness

Tests whether learner rankings hold across three DGPs designed to favor different learners (using tuned learners, n ∈ {200, 500, 1000}, 100 replications):

| DGP | Structure | Favorable learner |
|-----|-----------|-------------------|
| Sparse Linear | p=100, 5 active variables, linear nuisance | Lasso |
| Piecewise | p=20, indicator functions | Random Forest |
| Nonlinear Interaction | p=20, sin/tanh/exp interactions | Neural Network |

Learner-DGP mismatch causes RMSE to increase by 4-5x and CI coverage to drop well below 95%, confirming that DML's n^{-1/2} guarantee for θ̂ depends critically on nuisance estimation quality.

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
├── notebooks/             # demo and experiment notebooks
├── requirements.txt
└── setup.py
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

Key dependencies: `numpy`, `scikit-learn`, `torch`, `jax`, `econml`, `doubleml`, `matplotlib`, `pandas`.

---

## Reproducibility

All experiments use `random_state=66` as the default seed. Monte Carlo replications use `random_state=rep` (0-indexed) for both data generation and PLR/IRM fitting, ensuring each replication is independently reproducible.

---

## Running Tests

```bash
pytest tests/ -v
```

59 tests across 5 files covering learner API, PLR, IRM, cross-fitting utilities, and CausalForest.

---

## Running Experiments

```python
from experiments.exp2_learner_comparison import run_experiment_2, plot_experiment_2

df = run_experiment_2(n_reps=500)
plot_experiment_2(df, save_path='results/exp2_learner_comparison.png')
```

---

## Future Work

- Honest cross-fitting: compare `cross_fit_honest` vs standard `cross_fit` on coverage
- CausalForest honesty mechanism: test whether `honest=True` adds value within DML cross-fitting
- IRM benchmark with classifier-based propensity score estimation
- Orthogonality verification using true nuisance functions to isolate the score's mathematical property from estimation noise

---

## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.

Farrell, M. H., Liang, T., & Misra, S. (2021). Deep neural networks for estimation and inference. *Econometrica*, 89(1), 181–213.

Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML — An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6.