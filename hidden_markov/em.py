# https://github.com/sidravi1/Blog/blob/master/nbs/hmm_em.ipynb

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as sct
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from dgp import dgp_hmm

np.random.seed(42)

# matrix of transition probabilities
P = np.array([
    [0.7, 0.20, 0.10],
    [0.3, 0.6, 0.10 ],
    [0.1, 0.20, 0.70]
])
# parameters for emission normal distribution
mu = np.array([5.0, 0.0, -5.0])
sd = np.array([2.0, 0.5, 3.0])
# Number of observations to simulate
T = 1000

x, y, pi = dgp_hmm(P, mu, sd, T)

def forward_log_gaussian(y, init, P, mus, sds):
    """
    Log-space forward algorithm.

    Returns:
      loglik: float
      log_alpha: (T, K) log P(x_t | y_0:t)
    """
    y = np.asarray(y, dtype=float)
    init = np.asarray(init, dtype=float)
    P = np.asarray(P, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sds = np.asarray(sds, dtype=float)

    T = len(y)
    K = len(init)

    log_init = np.log(init)
    log_P = np.log(P)

    log_alpha = np.empty((T, K))

    # t = 0
    log_b0 = np.array([
        -0.5*np.log(2*np.pi*sds[k]**2) - 0.5*((y[0]-mus[k])**2)/(sds[k]**2)
        for k in range(K)
    ])
    log_alpha[0] = log_init + log_b0
    loglik = logsumexp(log_alpha[0])
    log_alpha[0] -= loglik

    # t = 1..T-1
    for t in range(1, T):
        log_bt = np.array([
            -0.5*np.log(2*np.pi*sds[k]**2) - 0.5*((y[t]-mus[k])**2)/(sds[k]**2)
            for k in range(K)
        ])

        log_pred = logsumexp(
            log_alpha[t-1][:, None] + log_P,
            axis=0
        )

        log_alpha[t] = log_pred + log_bt
        ct = logsumexp(log_alpha[t])
        log_alpha[t] -= ct
        loglik += ct

    return loglik, log_alpha

def backward_log_gaussian(y, P, mus, sds):
    y = np.asarray(y)
    P = np.asarray(P)
    mus = np.asarray(mus)
    sds = np.asarray(sds)

    T = len(y)
    K = len(mus)

    log_P = np.log(P)

    log_beta = np.zeros((T, K))   # log(1) at T-1

    for t in range(T-2, -1, -1):
        log_bt1 = np.array([
            -0.5*np.log(2*np.pi*sds[k]**2)
            -0.5*((y[t+1]-mus[k])**2)/(sds[k]**2)
            for k in range(K)
        ])

        for i in range(K):
            log_beta[t, i] = logsumexp(
                log_P[i] + log_bt1 + log_beta[t+1]
            )

    return log_beta

def smooth_log(log_alpha, log_beta):
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return log_gamma

def xi_log(y, log_alpha, log_beta, P, mus, sds):
    y = np.asarray(y)
    P = np.asarray(P)
    mus = np.asarray(mus)
    sds = np.asarray(sds)

    T, K = log_alpha.shape
    log_P = np.log(P)

    log_xi = np.empty((T-1, K, K))

    for t in range(T-1):
        log_bt1 = np.array([
            -0.5*np.log(2*np.pi*sds[k]**2)
            -0.5*((y[t+1]-mus[k])**2)/(sds[k]**2)
            for k in range(K)
        ])

        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = (
                    log_alpha[t, i]
                    + log_P[i, j]
                    + log_bt1[j]
                    + log_beta[t+1, j]
                )

        log_xi[t] -= logsumexp(log_xi[t])

    return log_xi

def m_step_gaussian(y, log_gamma, log_xi, min_sd=1e-3):
    """
    M-step for Gaussian HMM using log-space E-step outputs.

    Returns:
      init_new : (K,)
      P_new    : (K, K)
      mus_new  : (K,)
      sds_new  : (K,)
    """
    y = np.asarray(y, dtype=float)
    gamma = np.exp(log_gamma)
    xi = np.exp(log_xi)

    T, K = gamma.shape

    # ---- initial distribution ----
    init_new = gamma[0].copy()

    # ---- transition matrix ----
    xi_sum = xi.sum(axis=0)            # (K, K)
    gamma_sum = gamma[:-1].sum(axis=0) # (K,)
    P_new = xi_sum / gamma_sum[:, None]

    # ---- emission parameters ----
    w = gamma.sum(axis=0)              # (K,)

    mus_new = (gamma * y[:, None]).sum(axis=0) / w

    var_new = (gamma * (y[:, None] - mus_new[None, :])**2).sum(axis=0) / w
    sds_new = np.sqrt(np.maximum(var_new, min_sd**2))

    return init_new, P_new, mus_new, sds_new

def baum_welch_gaussian_log(
    y, K, init, P, mus, sds,
    n_iter=50, tol=1e-6
):
    init = np.asarray(init, dtype=float)
    P = np.asarray(P, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sds = np.asarray(sds, dtype=float)

    logliks = []

    for it in range(n_iter):

        # ---- E-step (log-space) ----
        loglik, log_alpha = forward_log_gaussian(y, init, P, mus, sds)
        log_beta = backward_log_gaussian(y, P, mus, sds)

        log_gamma = smooth_log(log_alpha, log_beta)
        log_xi = xi_log(y, log_alpha, log_beta, P, mus, sds)

        # ---- M-step ----
        init_new, P_new, mus_new, sds_new = m_step_gaussian(
            y, log_gamma, log_xi
        )

        logliks.append(loglik)

        # ---- convergence ----
        if it > 0 and abs(logliks[-1] - logliks[-2]) < tol:
            init, P, mus, sds = init_new, P_new, mus_new, sds_new
            break

        init, P, mus, sds = init_new, P_new, mus_new, sds_new

    return {
        "init": init,
        "P": P,
        "mus": mus,
        "sds": sds,
        "logliks": np.asarray(logliks),
    }

# call forward pass
K = 3
obs = y
# dists = (
#     sct.norm(loc=5, scale=2), 
#     sct.norm(loc=0, scale=0.5), 
#     sct.norm(loc=-5, scale=3)
# ) 
init = np.ones(K) / K
P = np.array([[0.6, 0.2, 0.2],
               [0.2, 0.6, 0.2],
               [0.2, 0.2, 0.6]])

mus = np.array([2.0, 0.0, -2.0])
sds = np.array([1.5, 1.0, 2.5])

fit = baum_welch_gaussian_log(
    obs, K, init, P, mus, sds, 
    n_iter=1000, tol=1e-6
    )

## sanity checks
# monotone log-likelihood
print(np.all(np.diff(fit["logliks"]) >= -1e-8))
# valid probabilities
np.allclose(fit["init"].sum(), 1.0)
np.allclose(fit["P"].sum(axis=1), 1.0)
# positive variances
np.all(fit["sds"] > 0)

print("fit_init", fit["init"])
print("fit_P",  fit["P"])
print("fit_mus", fit["mus"]) 
print("fit_sds", fit["sds"])

# run after em estimates parameters the forward pass and backward pass
# alpha_hat → filtered probabilities
# gamma_hat → smoothed probabilities

# Run smoothing with fitted parameters
loglik, log_alpha = forward_log_gaussian(
    y, fit["init"], fit["P"], fit["mus"], fit["sds"]
)
log_beta = backward_log_gaussian(
    y, fit["P"], fit["mus"], fit["sds"]
)
log_gamma = smooth_log(log_alpha, log_beta)
gamma = np.exp(log_gamma)

# Compare to true states (simulation only)
state_hat = gamma.argmax(axis=1)
accuracy = np.mean(state_hat == x)
print("State accuracy no permutation:", accuracy)

K = gamma.shape[1]
conf = np.zeros((K, K))

for i in range(K):
    for j in range(K):
        conf[i, j] = np.sum((x == i) & (state_hat == j))

row_ind, col_ind = linear_sum_assignment(-conf)
accuracy = conf[row_ind, col_ind].sum() / len(x)

print("State accuracy after permutation:", accuracy)


import matplotlib.pyplot as plt
plt.plot(fit["logliks"])
plt.xlabel("EM iteration")
plt.ylabel("log-likelihood")
plt.title("Baum–Welch convergence")
plt.show()

# t0, t1 = 0, 100
# t = np.arange(t0, t1 + 1)

# fig, ax = plt.subplots(figsize=(12,4))
# for k in range(alpha_hat.shape[1]):
#     ax.plot(t, alpha_hat[t0:t1+1, k], label=f"Filtered k={k}", alpha=0.7)
#     ax.plot(t, gamma_hat[t0:t1+1, k], ls="--", label=f"Smoothed k={k}", alpha=0.9)

# ax.set_ylim(0, 1)
# ax.set_xlabel("Time")
# ax.set_ylabel("Probability")
# ax.set_title("Filtered (solid) vs Smoothed (dashed)")
# ax.legend(ncol=3)
# plt.tight_layout()
# plt.show()

# comparison with hmmlearn
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    init_params="",
    params="stmc",
    n_iter=100,
    tol=1e-6,
    random_state=42
)

model.startprob_ = fit["init"]
model.transmat_ = fit["P"]
model.means_ = fit["mus"].reshape(-1, 1)
model.covars_ = (fit["sds"]**2).reshape(-1, 1)

model.fit(y.reshape(-1, 1))

hmm_init = model.startprob_
hmm_P = model.transmat_
hmm_mus = model.means_.ravel()
hmm_sds = np.sqrt(model.covars_.ravel())
hmm_loglik = model.score(y.reshape(-1, 1))

print("Your loglik:", fit["logliks"][-1])
print("hmmlearn loglik:", hmm_loglik)

# permutation
K = 3
cost = np.zeros((K, K))

for i in range(K):
    for j in range(K):
        cost[i, j] = abs(fit["mus"][i] - hmm_mus[j])

row, col = linear_sum_assignment(cost)
# permute hmmlearn estimates
hmm_P_perm = hmm_P[col][:, col]
hmm_mus_perm = hmm_mus[col]
hmm_sds_perm = hmm_sds[col]

print("Transition matrices:")
print("Yours:\n", fit["P"])
print("hmmlearn:\n", hmm_P_perm)
print("\nMeans:")
print("Yours:", fit["mus"])
print("hmmlearn:", hmm_mus_perm)
print("\nStds:")
print("Yours:", fit["sds"])
print("hmmlearn:", hmm_sds_perm)

post_hmm = model.predict_proba(y.reshape(-1, 1))[:, col]

# Agreement
state_hat_hmm = post_hmm.argmax(axis=1)
accuracy = np.mean(state_hat == state_hat_hmm)
print("Posterior state agreement:", accuracy)

# viterbi
def viterbi_log_gaussian(y, init, P, mus, sds):
    """
    Log-space Viterbi decoding for univariate Gaussian HMM.

    Returns:
      path: (T,) most likely state sequence
    """
    y = np.asarray(y, dtype=float)
    init = np.asarray(init, dtype=float)
    P = np.asarray(P, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sds = np.asarray(sds, dtype=float)

    T = len(y)
    K = len(init)

    log_init = np.log(init)
    log_P = np.log(P)

    # log emission densities
    logB = np.empty((T, K))
    for k in range(K):
        logB[:, k] = (
            -0.5*np.log(2*np.pi*sds[k]**2)
            -0.5*((y - mus[k])**2) / (sds[k]**2)
        )

    # DP tables
    delta = np.empty((T, K))     # best log prob up to t ending in k
    psi = np.empty((T, K), int)  # backpointers

    # initialization
    delta[0] = log_init + logB[0]
    psi[0] = 0

    # recursion
    for t in range(1, T):
        for j in range(K):
            vals = delta[t-1] + log_P[:, j]
            psi[t, j] = np.argmax(vals)
            delta[t, j] = vals[psi[t, j]] + logB[t, j]

    # termination
    path = np.empty(T, int)
    path[T-1] = np.argmax(delta[T-1])

    # backtracking
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path
    
vit_hmm = model.predict(y.reshape(-1, 1))
vit_hmm = col[vit_hmm]   # align labels
vit_you = viterbi_log_gaussian(y, fit["init"], fit["P"], fit["mus"], fit["sds"])
np.mean(vit_you == vit_hmm)