import numpy as np
# function to generate data
def dgp(mu: float, sigma: float, n: int):
    return np.random.normal(mu= mu, sigma=sigma, size=n)

# function to define an estimator
def estimator(data: np.ndarray):
    return np.mean(data)
    #return np.square(np.mean(data))

# function to apply the one-leave-out on dgp
def leave_one_out_data(data: np.ndarray):
    """
    Generate n leave-one-out datasets from a 1D NumPy array.
    
    Parameters:
        data (np.ndarray): A 1D array of data points.

    Returns:
        list of np.ndarray: A list of datasets, each missing one observation.
    """
    n = len(data)
    return [np.delete(data, i) for i in range(n)]

# function to apply the one-leave-out on estimator and compute average
def leave_one_out_estimator(data: np.ndarray):
    """
    Computes the leave-one-out estimate of an estimator.

    Parameters:
        data (np.ndarray): A 1D array of data points.

    Returns:
        float: The average of the leave-one-out estimator applied to each subset.
    """
    return np.mean([estimator(subset) for subset in leave_one_out_data(data)])

# bias estimation and bias-corrected jackknife estimator
def bias_jackknife_estimate(data: np.ndarray):
    """
    Computes the bias-corrected jackknife estimator and the jackknife estimate of bias.

    Parameters:
        data (np.ndarray): A 1D NumPy array of data points.

    Returns:
        dict: A dictionary containing:
            - 'theta_jack' (float): Bias-corrected jackknife estimate.
            - 'bias_jackknife' (float): Jackknife estimate of bias.
    """
    n = len(data)
    # Estimator on full data
    theta_hat = estimator(data)
    # Leave-one-out estimate  
    theta_hat_loo = leave_one_out_estimator(data)  
    # Bias-corrected estimate
    theta_jack = n * theta_hat - (n - 1) * theta_hat_loo
    # Jackknife estimate of bias
    bias_jackknife = (n - 1) * (theta_hat_loo - theta_hat)  
    
    return {
        "theta_jack": theta_jack,
        "bias_jackknife": bias_jackknife
    }

# variance estimation jackknife estimate
def variance_jackknife_estimate(data: np.ndarray):
    """
    Computes the jackknife estimate of variance.

    Parameters:
        data (np.ndarray): A 1D NumPy array of data points.

    Returns:
        float: The jackknife estimate of variance.
    """
    n = len(data)
    # Leave-one-out estimates
    jack_replication = np.array([estimator(subset) for subset in leave_one_out_data(data)]) 
    theta_hat_loo = leave_one_out_estimator(data)
    # Jackknife estimate of variance
    variance_jackknife = ((n - 1) / n) * np.sum((jack_replication - theta_hat_loo)**2)
    return variance_jackknife

# apply the code
data = dgp(mu=0, sigma=1, n=5000)
# mean and variance of data
print(f"Mean of data: {np.mean(data)}")
print(f"Variance of data: {np.var(data)}")
print(f"Estimator applied to dgp: {estimator(data)}")
# bias-corrected jackknife estimator
print(f"Bias-corrected jackknife estimator: {bias_jackknife_estimate(data)['theta_jack']}")
# jackknife estimate of bias
print(f"Jackknife estimate of bias: {bias_jackknife_estimate(data)['bias_jackknife']}")
# jackknife estimate of variance
print(f"Jackknife estimate of variance: {variance_jackknife_estimate(data)}")