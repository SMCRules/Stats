import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(42)
# simulate data generating process from Normal distribution
# population parameters:
mu = 1.5
# standard deviation -> square root of variance = 2
sigma = np.sqrt(2)
# sample size 
N = 1000
def simulate_data(mu, sigma, N):
    sample = np.random.normal(mu, sigma, N)
    return sample

# simulate data and check sample mean and variance
data = simulate_data(mu, sigma, N)
mu_plugin, sigma_plugin = np.mean(data), np.std(data)
print(f"Sample mean : {mu_plugin:.4f}")
print(f"Sample variance : {sigma_plugin**2:.4f}")

# plot histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='blue')
plt.xlabel('Samples')
plt.ylabel('Density')
plt.title('Histogram of Sample')
plt.show()

# the log-likelihood is a function of the unknown 
# population parameters mu and sigma
# y represents observed data

# Manual implementation looping through the univariate normal densities that make up the log-likelihood
# def log_likelihood(y, mu, sigma):
#     N = len(y)
#     cum_sum = 0
#     for i in range(N):
#         cum_sum += -0.5 * np.log(2 * np.pi) - np.log(sigma) - (y[i] - mu)**2 / (2 * sigma**2)
#     return cum_sum

# This version uses NumPy for vectorized version = much faster
def log_likelihood(y, mu, sigma):
    n = len(y)
    return -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - np.sum((y - mu) ** 2) / (2 * sigma**2)


# calculate log-likelihood for a range of values for mu [0, 3], 
# fixing sigma to sample variance (plug-in estimator)
mu_range = np.linspace(0, 3, 300)
grid_loglik = [log_likelihood(data, mu, sigma_plugin) for mu in mu_range]  # log_likelihood in mu_range]

# plot log-likelihood y-axis for each mu_range value on the x-axis
plt.figure(figsize=(10, 6))
plt.plot(mu_range, grid_loglik, linestyle='-', color='blue')
plt.xlabel('Mu (μ)')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood of Normal Distribution vs. μ')
plt.grid(True)
plt.show()

# To optimize mu and sigma using the log_likelihood function 
# use scipy's optimization package to minimize the negative log-likelihood. 
def neg_log_likelihood(params, y):
    mu, sigma = params
    if sigma <= 0:  # avoid log(0) or negative variance
        return np.inf
    n = len(y)
    return 0.5 * n * np.log(2 * np.pi) + n * np.log(sigma) + np.sum((y - mu) ** 2) / (2 * sigma ** 2)

# Initial guess for mu and sigma
initial_guess = [0, 1]

# Perform optimization
result = minimize(
    neg_log_likelihood, initial_guess, args=(data,), bounds=[(None, None), (1e-6, None)]
    )

# Extract estimated parameters
mle_mu, mle_sigma = result.x

print(f"MLE mu: {mle_mu:.4f}")
print(f"MLE sigma²: {mle_sigma**2:.4f}")
print(f"Sample mean (plug-in): {mu_plugin:.4f}")
print(f"Sample variance (plug-in): {sigma_plugin**2:.4f}")
