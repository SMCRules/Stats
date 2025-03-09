"""
the bootstrap estimates the (accuracy) sampling distribution of an estimator.
In the absence of any additional information about the estimator, 
the sample itself offers the best guide for the sampling distribution
"""
import numpy as np
import math

# function to generate data
def dgp(mu: float, sigma: float, n: int):
    """
    Generates n samples from a normal distribution with mean `mu` and standard deviation `sigma`.

    Parameters:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        n (int): Number of samples.

    Returns:
        np.ndarray: An array of `n` random samples.
    """
    return np.random.normal(loc=mu, scale=sigma, size=n)

# function to define an estimator
def estimator(data: np.ndarray):
    return np.mean(data)
    # return np.square(np.mean(data))

# Balanced bootstrap dataset results in B bootstrap samples, each containing
# B times the original data point.
def bootstrap_dataset(data: np.ndarray, B: int):
    """
    Generates a balanced bootstrap dataset. The B bootstrat sets are generated
    in such a way that each original data point is selected exactly B times in 
    the entire collection of bootstrap samples.

    Parameters:
        data (np.ndarray): A 1D array of data points.
        N (int): Number of points to select from the original dataset.

    Returns:
        np.ndarray: A 2D array of shape (B, len(data)) 
        containing bootstrap datasets.
    """
    data = np.asarray(data)  # Ensure it's a NumPy array
    N = len(data)
    
    # Generate B bootstrap samples with exactly B occurrences of each point
    bootstrap_samples = np.zeros((B, N), dtype=data.dtype)
    
    for i in range(B):
        bootstrap_samples[i] = np.random.permutation(data)
    
    return bootstrap_samples
    
# function to estimate bootstrap bias
def bootstrap_bias(data: np.ndarray, boot_samples: np.ndarray):
    """
    Computes the bootstrap estimate of the bias and the 
    bias-corrected estimate of an estimator.

    Parameters:
        data (np.ndarray): A 1D array of original data points.
        boot_samples (np.ndarray): A 2D array of shape (B, n) containing bootstrap samples.

    Returns:
        dict: A dictionary containing:
            - "boot_bias": The bootstrap estimate of the bias.
            - "boot_estimate": The bias-corrected estimate.
    """
    # Compute the average bootstrap estimate
    bootstrap_estimates = np.apply_along_axis(estimator, axis=1, arr=boot_samples)
    average_boot_estimator = np.mean(bootstrap_estimates)    
    # Compute the original estimate
    plugin_estimate = estimator(data)
    # Compute the bootstrap bias
    boot_bias = average_boot_estimator - plugin_estimate
    # Compute the bootstrap bias corrected estimate
    boot_estimate = (2 * plugin_estimate) - average_boot_estimator 
    
    return {

        "boot_bias": boot_bias,
        "boot_estimate": boot_estimate
    } 

# apply the code
data = dgp(mu=5, sigma=math.sqrt(10), n=5000)
# mean and variance of data
print(f"Mean of data: {np.mean(data)}")
print(f"Variance of data: {np.var(data)}")
print(f"Estimator applied to dgp: {estimator(data)}")

# Generate balanced bootstrap samples
B = 100
boot_samples = bootstrap_dataset(data, B)

# Compute bootstrap bias and corrected estimate
result = bootstrap_bias(data, boot_samples)

print("Bootstrap Bias:", result["boot_bias"])
print("Bias-Corrected Estimate:", result["boot_estimate"])
