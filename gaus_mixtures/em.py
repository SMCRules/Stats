import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import logsumexp

from dgp import dgp_gmm

# number of components mixture
K = 3
pi_k = [0.3, 0.5, 0.2]
mu_k = [-5, 0, 5]
sd_k = [3, 1, 3]

# number of observations
N = 10000

# generate data
x, y = dgp_gmm(N, K, pi_k, mu_k, sd_k)

def em_gmm(y, mu_init, sd_init, pi_init, n_iter=100, sd_min=1e-6):
    y = np.asarray(y)
    mu = np.asarray(mu_init, dtype=float).copy()
    sd = np.asarray(sd_init, dtype=float).copy()
    pi = np.asarray(pi_init, dtype=float).copy()

    K = len(pi)
    N = y.size

    for _ in range(n_iter):
        # E-step in log space
        log_weighted = np.zeros((K, N))
        for k in range(K):
            log_weighted[k, :] = (
                np.log(pi[k] + 1e-300) + stats.norm(mu[k], sd[k]).logpdf(y)
            )

        log_norm = logsumexp(log_weighted, axis=0)      # shape (N,)
        # ll = log_norm.sum()
        # print(_, ll, mu, sd, pi)
        
        r = np.exp(log_weighted - log_norm)             # shape (K, N)
        # M-step
        Nk = r.sum(axis=1) + 1e-300
        pi = Nk / N
        mu = (r @ y) / Nk
        # stable variance computation
        Ey2 = (r @ (y**2)) / Nk
        var = Ey2 - mu**2
        sd = np.sqrt(np.maximum(var, sd_min**2))
        # Optional but often helpful: ensure pi sums to 1 exactly
        pi = pi / pi.sum()

    return mu, sd, pi, r

### EM implementation
mu_init, sd_init, pi_init = [0, 0, 0], [1, 1, 1], [0.3, 0.5, 0.2]

mu_hat, sd_hat, pi_hat, r_hat = em_gmm(
    y, mu_init, sd_init, pi_init, n_iter = 10000
    )

order = np.argsort(mu_hat)
mu = mu_hat[order]
sd = sd_hat[order]
pi = pi_hat[order]
pi = pi / pi.sum()
r  = r_hat[order]   # IMPORTANT: keep responsibilities aligned
# compute hard labels
x_hat = np.argmax(r, axis=0)
# print out estimated parameters
for k in range(K):
    print(
        f"k={k}: "
        f"mu={mu[k]:.4f}, "
        f"sd={sd[k]:.4f}, "
        f"pi={pi[k]:.4f}"
    )

# Posterior mixture distribution
t = np.linspace(-15, 15, 1000)
y_mix = np.zeros_like(t)

for k in range(K):
    y_mix += pi[k]*stats.norm(mu[k], sd[k]).pdf(t)

fig = plt.figure(figsize=(8,4))
sns.histplot(x=y, stat='density', bins=100)
plt.plot(t,y_mix, color='r')
plt.xlabel('y')
plt.show()



