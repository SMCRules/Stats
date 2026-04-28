"""
-> We will assume a linear-Gaussian data generating process
this means the joint distribution of y,x is available to determine true risk analytically

1. Draw N training samples
2. Estimate a linear regression model y = beta * x
3. Compute true risk, empirical risk and bootstrap risk.
"""

import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# --- 1. True data-generating process ---
np.random.seed(0)
N = 5000
beta_true = 3.0
sigma = np.sqrt(5.0)

# Draw training data
# inputs x generated from N(0,1)
x = np.random.normal(0, 1, N)
# observations generated from N(Beta * x, sigma^2 = 2)
y = beta_true * x + np.random.normal(0, sigma, N)

# --- 2. Fit model (linear regression without intercept) ---
# theta_hat = (x'y) / (x'x)
theta_hat = np.sum(x * y) / np.sum(x ** 2)
print("theta_hat: ", theta_hat)
# --- 3. Compute True Risk analytically ---
true_risk = (theta_hat - beta_true) ** 2 * np.var(x) + sigma ** 2
print("true_risk: ", true_risk)
# --- 4. Compute Empirical Risk ---
y_pred_train = theta_hat * x
empirical_risk = np.mean((y - y_pred_train) ** 2)
print("empirical_risk: ", empirical_risk)
# --- 5. Bootstrap estimate of True Risk ---
"""
The goal is to estimate the “true risk” (expected prediction error on new data) 
using bootstrap resampling — i.e., repeatedly resampling the training data, 
fitting a model, and evaluating it on the points not used in training 
(the “out-of-bag” samples).
"""
B = 1000
bootstrap_oob_risks = []

# set deterministic indices outside loop
indices = np.arange(N)
for _ in range(B):
    # Sample indices with replacement    
    boot_indices = resample(indices, replace=True, n_samples=N)
    oob_indices = np.setdiff1d(indices, np.unique(boot_indices))

    # Fit model on bootstrap sample = training data
    x_boot, y_boot = x[boot_indices], y[boot_indices]
    theta_boot = np.sum(x_boot * y_boot) / np.sum(x_boot ** 2)

    # Evaluate on OOB samples (if any) = test "unseen" data
    if len(oob_indices) > 0:
        x_oob, y_oob = x[oob_indices], y[oob_indices]
        y_pred_oob = theta_boot * x_oob
        oob_risk = np.mean((y_oob - y_pred_oob) ** 2)
        bootstrap_oob_risks.append(oob_risk)

bootstrap_risk = np.mean(bootstrap_oob_risks)

# --- 6. Print results ---
print(f"True parameter beta_true = {beta_true:.2f}")
print(f"Fitted parameter theta_hat = {theta_hat:.2f}\n")
print(f"True Risk (analytical): {true_risk:.4f}")
print(f"Empirical Risk (training): {empirical_risk:.4f}")
print(f"Bootstrap OOB Risk Estimate: {bootstrap_risk:.4f}")

# --- 7. Visualize ---
plt.hist(bootstrap_oob_risks, bins=30, alpha=0.7)
plt.axvline(true_risk, color='r', linestyle='--', label='True risk')
plt.axvline(empirical_risk, color='g', linestyle='--', label='Empirical risk')
plt.axvline(bootstrap_risk, color='b', linestyle='--', label='Bootstrap mean')
plt.title('Bootstrap distribution of risk estimates')
plt.legend()
plt.xlabel('Out-of-bag MSE')
plt.ylabel('Frequency')
plt.show()

"""
Lets implement gradient descent algorithm and estimate Beta. 
We know that the true Beta value is 3.0. 
We have a model with a single parameter. 
We will use a risk function for a linear model with a squared-loss function.
x: will be the vector of inputs
y: will be the vector of dependent variable
"""

# Define functions for performing gradient descent
def risk(x, y, beta):
    '''
    Function to compute the empirical risk
    
    :param x: The feature values for our example
    :type x: A 1D numpy array
    
    :param y: The response values for our example
    :type y: A 1D numpy array
    
    :param beta: The model parameter value at which we want to 
                 evaluate the empirical risk
    :type beta: float
    
    :return: The empirical risk function value
    :rtype: float
    '''
    # Initialize the risk value
    risk = 0.0
  
    # Loop over the data and increment the risk with 
    # a squared-loss
    for i in range(x.shape[0]):
        risk += np.power(y[i]-(beta*x[i]), 2.0)
        
    risk /= x.shape[0]
  
    return risk


def derivative_risk(x, y, beta):
    '''
    Function to compute the derivative of the empirical risk
    with respect to the model parameter
    
    :param x: The feature values for our example
    :type x: A 1D numpy array
    
    :param y: The response values for our example
    :type y: A 1D numpy array
    
    :param beta: The model parameter value at which we want to 
                 evaluate the empirical risk
    :type beta: float
    
    :return: The derivative of the empirical risk function
    :rtype: float
    '''
    derivative_risk = 0.0
  
    for i in range(x.shape[0]):
        derivative_risk += - (2.0*x[i]*(y[i]-(beta*x[i])))

    derivative_risk /= x.shape[0]
    
    return derivative_risk   

# Set the learning rate and the number of iterations we want to perform
eta=0.05
n_iter=200

# Initialize arrays to hold the sequence of 
# parameter estimates and empirical risk values
beta_learn=np.full(1+n_iter, np.nan)
risk_learn=np.full(1+n_iter, np.nan)

# Set the starting estimate for the
# model parameter
beta_learn[0]=1.0

# Iterate using the gradient descent update rule
for iter in range(n_iter):
    risk_learn[iter] = risk(x,y,beta_learn[iter])
    beta_learn[iter+1] = beta_learn[iter]
    beta_learn[iter+1] -= (eta*derivative_risk(x,y,beta_learn[iter]))

# Plot parameter estimates at each iteration
plt.plot(beta_learn, marker="o")
plt.title(r'$\hat{\beta}$ vs Iteration', fontsize=24)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel(r'$\hat{\beta}$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Plot empirical risk at each iteration
plt.plot(risk_learn, marker="o")
plt.title('Empirical risk vs Iteration', fontsize=24)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Empirical risk', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# a log-likelihood–based version of squared loss is a Normal log-likelihood
def log_likelihood(x, y, beta, sigma):
    N = x.shape[0]
    residuals = y - beta * x
    ll = -0.5 * N * np.log(2 * np.pi * sigma**2) \
         - np.sum(residuals**2) / (2 * sigma**2)
    return ll

def d_log_likelihood(x, y, beta, sigma):
    residuals = y - beta * x
    d_ll = np.sum(x * residuals) / (sigma**2)
    return d_ll

# gradient ascent (because we’re maximizing likelihood):
eta=0.001
risk_ml = np.full(1+n_iter, np.nan)
beta_ml = np.full(1+n_iter, np.nan)
beta_ml[0] = 1.0
for iter in range(n_iter):
    risk_ml[iter] = log_likelihood(x, y, beta_ml[iter], sigma)
    grad = d_log_likelihood(x, y, beta_ml[iter], sigma)
    beta_ml[iter+1] = beta_ml[iter] + (eta * grad)  # <-- ascent direction


# Plot parameter estimates at each iteration
plt.plot(beta_ml, marker="o")
plt.title(r'$\hat{\beta}$ vs Iteration', fontsize=24)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel(r'$\hat{\beta}$', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# Plot empirical risk at each iteration
plt.plot(risk_ml, marker="o")
plt.title('Empirical risk vs Iteration', fontsize=24)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Empirical risk', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
