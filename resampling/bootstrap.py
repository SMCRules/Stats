"""
the bootstrap estimates the (accuracy) sampling distribution of an estimator.
In the absence of any additional information about the estimator, 
the sample itself offers the best guide for the sampling distribution
"""
import numpy as np

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
    
data = list(range(1, 11))
print(data)
bootstrap_samples = bootstrap_dataset(data, B=5)
print(bootstrap_samples)



