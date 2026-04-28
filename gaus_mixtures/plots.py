import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats


# Plots DGP observations
fig = plt.figure(figsize=(8,4))
sns.histplot(x=y, bins=100, kde=True)
plt.axvline(x=-5, color='red', alpha=0.6, linewidth=2)
plt.axvline(x=0, color='red', alpha=0.6, linewidth=2)
plt.axvline(x=5, color='red', alpha=0.6, linewidth=2)
plt.xlabel('Observations', fontsize=14, fontweight='bold', color='gray')
plt.show()

#### Plots posterior probabilities
# plot posterior probabilities = soft assignment
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(x=y, bins=100, kde=True, ax=ax)
ax.vlines(y, 0, 50, color='orange', alpha=px_1)
ax.vlines(y, 0, 50, color='steelblue', alpha=px_2)
ax.vlines(y, 0, 50, color='green', alpha=px_3)
plt.xlabel("y")
plt.show()

# Scatter plot with color blending (posterior mixing)
fig, ax = plt.subplots(figsize=(8,4))
colors = np.stack([px_1, px_2, px_3], axis=1)
ax.scatter(
    y,
    np.zeros_like(y),
    c=colors,
    s=20,
    alpha=0.7
)
ax.set_yticks([])
ax.set_xlabel("y")
plt.show()

# Posterior curves over x (clean & analytical)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(y, px_1, label="P(z=1|y)", color="orange")
ax.plot(y, px_2, label="P(z=2|y)", color="steelblue")
ax.plot(y, px_3, label="P(z=3|y)", color="green")
ax.set_xlabel("y")
ax.set_ylabel("posterior probability")
ax.legend()
plt.show()

# Stacked posterior density (advanced, compact)
idx = np.argsort(y)
y_sorted = y[idx]
fig, ax = plt.subplots(figsize=(8,4))
ax.stackplot(
    y_sorted,
    px_1[idx],
    px_2[idx],
    px_3[idx],
    labels=["z=1", "z=2", "z=3"],
    colors=["orange", "steelblue", "green"],
    alpha=0.6
)
ax.set_xlabel("y")
ax.set_ylabel("posterior mass")
ax.legend()
plt.show()

# Posterior mixture distribution
t = np.linspace(-15, 15, 1000)
y_mix = np.zeros_like(t)

for k in range(K):
    y_mix += pi_hat[k]*stats.norm(mu_hat[k], sd_hat[k]).pdf(t)

fig = plt.figure(figsize=(8,4))
sns.histplot(x=y, stat='density', bins=100)
plt.plot(t,y_mix, color='r')
plt.xlabel('y')
plt.show()
