# https://nicolas-hug.com/blog/gradient_boosting_descent

"""
estimation of univariate theta for linear regression y = theta * x
with gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt

def L2_loss(y_true, y_pred):
    """
	L2 loss (mean of squared errors).
	"""
    return np.mean((y_true - y_pred)**2)

def L1_loss(y_true, y_pred):
    """
    L1 loss (mean of absolute difference)
    """
    return np.mean(np.abs(y_true - y_pred))

def gradient_L2(X, y_true, y_pred):
    """
    Return gradient wrt beta for L2 loss (MSE):
    grad = np.sum(-2 * (y_true - y_pred) * X)/len(y)
    """
    grad = 2.0 * np.sum( X * (y_pred - y_true))
    return grad / len(y_true)

def gradient_L1(X, y_true, y_pred):
    # gradient w.r.t parameters theta
    sign_vec = np.sign(y_true - y_pred)
    return - X.T.dot(sign_vec) / len(y_true)
    
 
# simulate data y = theta * x + noise
n_samples = 3000
n_iterations = 500
lr = .01
# model parameters, beta and sigma
beta_true = 3.5
sigma = np.sqrt(2.0)

# generate data
np.random.seed(42)
X = np.random.randn(n_samples)
noise = np.random.randn(n_samples) * sigma
noise_var = np.var(noise)
# model
y_true = (beta_true * X) + noise
# mean and variance of observations y_true
y_mean = np.mean(y_true)
print("y mean:", y_mean)
y_var = np.var(y_true)
print("y variance:", y_var)

# store beta, loss and gradient
store_beta = []
store_loss = []
store_gradient = []
prop_beta = 0.0

for m in range(n_iterations):
    y_pred = X * prop_beta
    loss = L2_loss(y_true, y_pred)
    negative_gradient = -gradient_L2(X, y_true, y_pred, average=True)
    prop_beta += lr * negative_gradient

    store_beta.append(prop_beta)
    store_loss.append(loss)
    store_gradient.append(negative_gradient)

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# ---- 1. Beta values
axes[0].plot(store_beta, color='tab:blue')
axes[0].axhline(y=beta_true, color='tab:red', linestyle='--', label='True β')
axes[0].set_ylabel(r'$\beta$ estimate', fontsize=12)
axes[0].set_title('Gradient Descent Progress', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)

# ---- 2. Loss
axes[1].plot(np.log(store_loss), color='tab:orange')
axes[1].set_ylabel('log Loss', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)

# ---- 3. Gradient magnitude
axes[2].plot(store_gradient, color='tab:green')
axes[2].set_ylabel('Gradient', fontsize=12)
axes[2].set_xlabel('Iteration', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# lets estimate the standard error of beta
print("Proposed beta:", prop_beta)
y_pred = X * prop_beta
residuals = y_true - y_pred

# Estimate sigma^2 (unbiased)
rss = np.sum(residuals**2)
sigma2_hat = rss / (len(y_true) - 1)
print("Estimated sigma^2:", sigma2_hat)

# Covariance matrix of beta
# reshape the X
X = X.reshape(-1, 1)
XtX = X.T @ X
cov_beta = sigma2_hat * np.linalg.inv(XtX)

# Standard errors
se_beta = np.sqrt(np.diag(cov_beta))

print("Covariance matrix of beta:\n", cov_beta)
print("Standard errors:", se_beta)
