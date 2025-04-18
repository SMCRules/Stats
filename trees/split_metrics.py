import numpy as np
import matplotlib.pyplot as plt


def gini(probas):
    return np.array([1- (p ** 2 + (1-p) ** 2) for p in probas])


def entropy(probas):
    return np.array([
        -1 * (p * np.log2(p) + (1-p) * np.log2(1-p)) for p in probas
        ]
        )


def misclass_error_rate(probas):
    return np.array([1 - max([p, 1-p]) for p in probas])


probas = np.linspace(0.001, 0.999, 250)
plt.plot(probas, entropy(probas), label="Shannon's entropy")
plt.plot(probas, 2 * gini(probas),  label="Gini impurity x 2")
plt.plot(probas, 2 * misclass_error_rate(probas), label="Misclass error x 2")
plt.plot(probas, gini(probas), label="Gini impurity")
plt.plot(probas, misclass_error_rate(probas), label="Misclass error")
plt.title("Splitting criteria from P+ (binary classification case)")
plt.xlabel("P+")
plt.ylabel("Impurity")
plt.legend()
plt.show()