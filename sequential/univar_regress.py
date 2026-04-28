
"""
This is a simple univariate regression model y = 2 * X + np.random.normal(0, 1, Nobs)
which is going to be estimated in batch with OLS and in sequential with SGD.

The output will be two plots 
- one for the error over time and 
one for the weight over time
"""

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
random.seed(42)

# ----------------------
# Batch Learning
# ----------------------
class Batch:
    def train_model(self, X, y):
        # closed form solution for slope (no intercept)
        w = np.sum(X * y) / np.sum(X * X)
        return w

# ----------------------
# Sequential Learning (Online Stochastic Gradient Descent)
# ----------------------
class Sequential: 
    def __init__(self, decay = None):
        self.weight = 0.0
        self.decay = decay 

    def predict(self, x):
        return self.weight * x

    def update(self, x, y, lr=0.01):
        pred = self.predict(x)
        error = y - pred

        if self.decay is not None:
            # learning rate decay
            self.weight = self.decay * self.weight + lr * error * x
        else:
            # constant learning rate
            self.weight += lr * error * x
        
        return error

# ----------------------
# Data Generator (stream)
# ----------------------
def data_stream(X, y):
    for x, y in zip(X, y):
        yield x, y

# ----------------------
# Online Learning Loop
# ----------------------
def online_learning_system(model, stream):
    errors, weights = [], []

    for step, (x, y) in enumerate(stream):
        pred = model.predict(x)
        error = model.update(x, y)

        errors.append(round(abs(error), 2))
        weights.append(round(model.weight, 2))

        # print({
        #     "step": step,
        #     "x": round(x, 2),
        #     "prediction": round(pred, 2),
        #     "true": round(y, 2),
        #     "error": round(error, 2),
        #     "weight": round(model.weight, 3)
        # })

    return errors, weights

def plot_errors(errors):
    plt.figure()
    plt.plot(errors)
    plt.title("Error Over Time (Sequential Learning)")
    plt.xlabel("Step")
    plt.ylabel("Absolute Error")
    plt.show()

def plot_weights(weights, batch_weight=None):
    plt.figure()
    plt.plot(weights, label="Sequential Weight")

    if batch_weight is not None:
        plt.axhline(batch_weight, linestyle='--', label="Batch Weight")

    plt.title("Weight Convergence")
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()



# ----------------------
# Window-based smoothing
# ----------------------
class WindowEstimator:
    def __init__(self, size=5):
        self.window = deque(maxlen=size)

    def update(self, x, y):
        self.window.append((x, y))
        return np.mean([y for _, y in self.window])

# ----------------------
# Run everything
# ----------------------
if __name__ == "__main__":
    # Generate dataset
    Nobs = 500 
    X = np.random.uniform(0, 10, Nobs)
    y = 2 * X + np.random.normal(0, 1, Nobs)

    # Batch model
    batch = Batch()
    w_batch = batch.train_model(X, y)
    print("Batch weight:", round(w_batch, 3))

    # Sequential model
    seq_model = Sequential()
    stream = data_stream(X, y)
    # print("\n--- Online Learning ---")
    # online_learning_system(seq_model, stream)
    # print("Sequential weight:", seq_model.weight)

    errors, weights = online_learning_system(seq_model, stream)

    plot_errors(errors)
    plot_weights(weights, batch_weight=w_batch)

