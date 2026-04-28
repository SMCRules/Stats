# the purpose of this file is to 
# implement linear regression estimation, mle and introduce regularization
# in two datasets simulated and housing dataset

import pandas as pd
import numpy as np

def loglik_linear(y, X, Beta, sigma2):
    T = y.shape[0]
    resid = y - X @ Beta
    rss = resid.T @ resid  # scalar
    ll = -0.5 * T * np.log(2 * np.pi * sigma2) - 0.5 / sigma2 * rss
    return ll

def make_linear_regression(
    n_samples=1000, 
    n_informative=5, 
    n_redundant=3, 
    n_borderline=2, 
    noise=1.0, 
    random_state=None
    ):
    rng = np.random.default_rng(random_state)
    p = n_informative + n_redundant + n_borderline

    # --- Step 0: intercept ---
    intercept = 5.0 
    
    # --- Step 1: generate informative predictors ---
    X_inf = rng.normal(size=(n_samples, n_informative))
    
    # true coefficients for informative predictors
    beta_inf = rng.uniform(-3, 3, size=n_informative) 
    
    # --- Step 2: redundant predictors (linear combos) ---
    coeffs_redundant = rng.uniform(-1, 1, size=(n_informative, n_redundant))
    X_red = X_inf @ coeffs_redundant   # exact collinearity
    
    # --- Step 3: borderline predictors (noisy linear combos) ---
    coeffs_border = rng.uniform(-1, 1, size=(n_informative, n_borderline))
    X_bor = X_inf @ coeffs_border + rng.normal(scale=0.1, size=(n_samples, n_borderline))  # almost collinear
    
    # --- Step 4: final design matrix (with constant column) ---
    X_ones = np.ones((n_samples, 1))
    X = np.hstack([X_ones, X_inf, X_red, X_bor])
    
    # --- Step 5: generate y (linear model with Gaussian noise) ---
    # Only informative features matter in truth
    y = intercept + X_inf @ beta_inf + rng.normal(scale=noise, size=n_samples)
    
    beta_true = np.hstack([intercept, beta_inf])
    return X, y, beta_true

X, y, beta_true = make_linear_regression(
    n_samples=5000, 
    n_informative=5, 
    n_redundant=3, 
    n_borderline=2, 
    noise=2.0, 
    random_state=42)

print("True informative betas:", beta_true.round(2))
# True informative betas: [5. -0.7  -1.9   1.51  0.94  2.67]
print("X shape:", X.shape)
# X shape: (5000, 11)
# fitting
from sklearn.linear_model import LinearRegression, Ridge, Lasso
# OLS (same as MLE for Gaussian linear regression)
ols_scratch = np.linalg.inv(X.T @ X) @ X.T @ y
print("OLS scratch coefficients:", ols_scratch.round(2))
"""
OLS scratch coefficients: [-8.07671070e+11  8.57284835e+12  1.92994579e+12 -9.25869685e+13
 -1.41777905e+13 -8.12333945e+11  3.62079240e+12 -1.16611388e+12
  3.59800754e+13 -9.40000000e-01  3.10000000e-01]
"""
# something is wrong here in ols_scratch
# diagnose by checking np.linalg.matrix_rank(X) and np.linalg.cond(X).
cond = np.linalg.cond(X)
cond_xtx = np.linalg.cond(X.T @ X)
rank = np.linalg.matrix_rank(X)
print("cond(X) =", cond)
print("cond(X^T X) =", cond_xtx)
print("rank(X) =", rank, " / ", X.shape[1])
# Very large cond (e.g. > 1e12) or rank < n_cols means trouble.

"""
Quick interpretation
rank(X) = 8 / 11 → 3 columns are linear combinations of others 
(exactly by construction: dgp redundant columns).
cond(X) ≫ 1e12 → matrix is numerically singular / 
ill-conditioned; inverting X.T@X is unreliable.

Safe replacements for inv(X.T@X) @ X.T @ y
1) Pseudo-inverse (SVD stable minimum-norm solution)
beta_pinv = np.linalg.pinv(X) @ y
2) lstsq (numerically stable)
beta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
"""
beta_pinv = np.linalg.pinv(X) @ y
print("pinv:", beta_pinv.round(3))
# pinv: [4.958 -0.355 -1.95   0.549  0.466  2.394  0.469  0.23   0.792  0.225  0.114]

beta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
print("lstsq:", beta_lstsq.round(3))
# lstsq: [ 4.958 -0.355 -1.95   0.549  0.466  2.394  0.469  0.23   0.792  0.225 0.114] 
"""
In practice:
Use lstsq if you also want the residuals, rank, and singular values back.
Use pinv if you just want the coefficients quickly.
"""
# lets try beta ridge
beta_ridge = np.linalg.solve(X.T @ X + 1e-6*np.eye(X.shape[1]), X.T @ y)
print("ridge:", np.round(beta_ridge,2))
# ridge: [ 4.958 -0.355 -1.95  0.549  0.466  2.394  0.469  0.23   0.792  0.225 0.114]
# If including a ones column in X, we need to tell sklearn not to fit intercept twice:
# model = LinearRegression(fit_intercept=False).fit(X, y)
ols = LinearRegression(fit_intercept=False).fit(X, y)
print("OLS coefficients:", ols.coef_.round(2))
# OLS coefficients: [ 4.96 -0.36 -1.95  0.55  0.47  2.39  0.47  0.23  0.79  0.22  0.11]

# Ridge (L2 regularization, MLE + penalty)
ridge = Ridge(fit_intercept=False, alpha=10).fit(X, y)
print("Ridge coefficients:", ridge.coef_.round(2))
# Ridge coefficients: [ 4.95 -0.32 -1.95  0.56  0.4   2.3   0.49  0.28  0.73  0.41  0.1 ]

# Lasso (L1 regularization, sparse solution)
lasso = Lasso(fit_intercept=False, alpha=0.1).fit(X, y)
print("Lasso coefficients:", lasso.coef_.round(2))
# Lasso coefficients: [ 4.86  0.   -1.92  0.26  0.    2.48  0.2  -0.    1.25  0.   -0.  ]

# mle estimation
# mle = 