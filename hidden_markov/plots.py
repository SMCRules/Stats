
import numpy as np
import matplotlib.pyplot as plt
from dgp import dgp_hmm



P = np.array([
    [0.7, 0.20, 0.10],
    [0.25, 0.6, 0.15],
    [0.1, 0.20, 0.70]
])
mu = np.array([5.0, 0.0, -5.0])
sd = np.array([2.0, 0.5, 3.0])
T = 1000

x, y, _ = dgp_hmm(P, mu, sd, T)
# plot
# choose a window so it’s readable
t0, t1 = 0, 300
t = np.arange(t0, t1 + 1)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 2]}
)

# ---- Top: discrete state path ----
ax1.step(t, x[t0:t1+1], where="post")
ax1.set_yticks([0, 1, 2])
ax1.set_ylabel("State")
ax1.set_title("Hidden state path")

# ---- Observations ----
ax2.plot(t, y[t0:t1+1])
ax2.set_ylabel("Observation")
ax2.set_xlabel("Time")
ax2.set_title("Simulated observations")


plt.tight_layout()
plt.show()
