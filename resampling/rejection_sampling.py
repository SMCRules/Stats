# lets practice control flow with a rejection sampling algorithm
# https://jaketae.github.io/study/rejection-sampling/
# https://cosmiccoding.com.au/tutorials/rejection_sampling/
# series of good tutorials:
# https://cosmiccoding.com.au/tutorials/
# https://medium.com/@roshmitadey/rejection-sampling-sampling-from-difficult-distributions-dbd17742a919
# is the medium algorithm correct?


from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def p(x):    
    """
    Target density.
    Mixture of two normal distributions.
    The first component is centered at 30 with standard deviation of 10.
    The second component is centered at 80 with standard deviation of 20.
    """
    return norm.pdf(x, loc=30, scale=10) + norm.pdf(x, loc=80, scale=20)
def q(x):
    """
    Proposal density.
    A normal distribution centered at 50 with standard deviation of 30.
    """
    return norm.pdf(x, loc=50, scale=30)

x = np.arange(-50, 151)
fig, ax = plt.subplots()
ax.plot(x, p(x), label=r"$p(x)$")
ax.plot(x, q(x), label=r"$q(x)$")
plt.legend()
plt.show()

# scale the proposal distribution
k = max(p(x) / q(x))

fig, ax = plt.subplots()
ax.plot(x, p(x), label=r"$p(x)$")
ax.plot(x, k * q(x), label=r"$k \cdot q(x)$")
plt.title("A containing proposal M * q(x)")
plt.legend()
plt.show()

acc_size = 10000
acc_count = 0
pro_count = 0
acc_store = []

while acc_count < acc_size:
    pro_count += 1
    # draw proposal
    sample = np.random.normal(50, 30)
    # draw uniform random number
    u = np.random.uniform(0, 1)
    # use constant M = max(p(x) / q(x))
    M = k
    # compute ratio
    ratio = p(sample)/(M*q(sample))
    # decide whether to accept or reject
    if u <= ratio:
        acc_count += 1
        acc_store.append(sample)
else:
    print("The acceptance ratio is", acc_count / pro_count)    

sns.displot(acc_store)
plt.title("Slow loop function")
plt.show()

def vector_rs(acc_size):
    samples = np.random.normal(50, 30, size=acc_size)
    u = np.random.uniform(0, 1, size=acc_size)
    mask = u <= p(samples)/(M*q(samples))
    return samples[mask]

acc_store = vector_rs(acc_size)
sns.displot(acc_store)
plt.title("vectorized rejection sampling function")
plt.show()