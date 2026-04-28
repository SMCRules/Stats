# https://github.com/tsmatz/hmm-lds-em-algorithm/blob/master/01-hmm-em-algorithm.ipynb

"""
Generate sample data from a Hidden Markov Model (HMM) 
"""
import numpy as np

np.random.seed(42)

def dgp_hmm(P, mu, sd, T):
    """
    Simulate data from a Hidden Markov Model
    inputs:
        P: 3x3 matrix of transition probabilities
        mu: scalar mean for emission normal distribution
        sd: scalar sd for emission normal distribution
        T: Number of observations to simulate
    outputs:
        x: univariate discrete state sequence
        y: univariate observation sequence
    """

    # Number of hidden states
    K = P.shape[0]
    # Stationary distribution for initialization π=πP
    eigvals, eigvecs = np.linalg.eig(P.T)
    pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    pi = pi / pi.sum()
    pi = np.maximum(pi, 0)   # guard against tiny negative numerical noise
    pi = pi / pi.sum()

    # create discrete state sequence x
    x = np.empty(T + 1, dtype=int)
    x[0] = np.random.choice(K, p=pi)

    # create observation sequence y 
    y = np.empty(T + 1)
    y[0] = np.random.normal(loc=mu[x[0]], scale=sd[x[0]])

    for t in range(T):
        # simulate discrete states
        x[t+1] = np.random.choice(K, p=P[x[t]])

        # simulate observation given updated state x[t+1] that indexes the normal density 
        y[t+1] = np.random.normal(loc=mu[x[t+1]], scale=sd[x[t+1]])

    return x, y, pi
