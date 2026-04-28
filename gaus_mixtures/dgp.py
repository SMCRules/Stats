# https://boyangzhao.github.io/posts/expectation-maximization

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats

np.random.seed(42)
def dgp_gmm(N, K, pi_k, mu_k, sd_k):
    x = np.empty(N, dtype=int)
    y = np.empty(N)

    for n in range(N):
        x[n] = np.random.choice(K, p=pi_k)
        y[n] = stats.norm.rvs(loc=mu_k[x[n]], scale=sd_k[x[n]])

    return x, y

# x, y = dgp_gmm(N, K, pi_k, mu_k, sd_k)

# # Mimics Expectation step
# # with known \theta = (pi_k, mu_k, sd_k) known, we can estimate 
# # the posterior probability P(x=k|y) = P(y|x=k)p(x=k).
# log_pdfs = np.array([
#     np.log(pi_k[k]) + stats.norm(mu_k[k], sd_k[k]).logpdf(y)
#     for k in range(len(pi_k))
# ])

# log_norm = scipy.special.logsumexp(log_pdfs, axis=0)
# responsibilities = np.exp(log_pdfs - log_norm) 
# px_1, px_2, px_3 = responsibilities
# # sanity check
# np.allclose(px_1 + px_2 + px_3, 1.0)
# # to obtain labels from posterior probabilities = hard assignment
# x_hat = np.argmax([px_1, px_2, px_3], axis=0)

# # Maximization step
# # estimate parameters
# # With known z, we can reconstitute the probability density function

# # with known labels, we can estimate the model params, 
# # by allocating observations to the correct density
# K = np.unique(x).size
# N = len(y)

# mu_hat = np.zeros(K)
# sd_hat = np.zeros(K)
# pi_hat = np.zeros(K)

# for k in range(K):
#     yk = y[x == k]

#     mu_hat[k] = yk.mean()
#     sd_hat[k] = yk.std(ddof=0)
#     pi_hat[k] = len(yk) / N

# for k in range(K):
#     print(
#         f"k={k}: "
#         f"mu={mu_hat[k]:.4f}, "
#         f"sd={sd_hat[k]:.4f}, "
#         f"pi={pi_hat[k]:.4f}"
#     )

