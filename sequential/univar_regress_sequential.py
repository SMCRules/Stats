"""
This is a continuation of univar_regress.py. 
Now we are adding a bias term. 
We are adding learning diagnostics. 
Concept drift simulations.
Visual comparisons. 
Batch vs sequential
Decreasing learning rate

Without drift:
both methods converge similarly
decay slightly worse (forgets useful info)
With drift:
no decay → stuck between 2 and 5
decay → adapts toward 5
👉 This is the core insight of online learning.
"""
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Models
# ----------------------
class Sequential:
    def __init__(self, decay=None):
        self.w = 0.0
        self.b = 0.0
        self.decay = decay

    def predict(self, x):
        return self.w * x + self.b

    def update(self, x, y, step, base_lr=0.01):
        # decreasing learning rate
        lr = base_lr / (1 + step * 0.01)

        pred = self.predict(x)
        error = y - pred

        if self.decay is not None:
            self.w = self.decay * self.w + lr * error * x
            self.b = self.decay * self.b + lr * error
        else:
            self.w += lr * error * x
            self.b += lr * error

        return error


class Batch:
    def train_model(self, X, y):
        # closed-form with bias
        X_design = np.vstack([X, np.ones(len(X))]).T
        w, b = np.linalg.lstsq(X_design, y, rcond=None)[0]
        return w, b


# ----------------------
# Data (with drift option)
# ----------------------
def generate_data(N=100, drift=False):
    X = np.random.uniform(0, 10, N)
    y = []

    for t, x in enumerate(X):
        noise = np.random.normal(0, 1)

        if drift:
            if t < N // 2:
                y.append(2 * x + noise)
            else:
                y.append(5 * x + noise)
        else:
            y.append(2 * x + noise)

    return X, np.array(y)


def data_stream(X, y):
    for xi, yi in zip(X, y):
        yield xi, yi


# ----------------------
# Online learning system
# ----------------------
def online_learning_system(model, stream):
    errors = []
    weights = []
    biases = []

    for step, (x, y) in enumerate(stream):
        pred = model.predict(x)
        error = model.update(x, y, step)

        errors.append(abs(error))
        weights.append(model.w)
        biases.append(model.b)

    return errors, weights, biases


# ----------------------
# Diagnostics
# ----------------------
def rolling_average(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_errors(errors_dict):
    plt.figure()
    for label, errors in errors_dict.items():
        plt.plot(rolling_average(errors), label=label)

    plt.title("Error Over Time (Smoothed)")
    plt.xlabel("Step")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.show()


def plot_weights(weights_dict, batch_w=None):
    plt.figure()
    for label, weights in weights_dict.items():
        plt.plot(weights, label=label)

    if batch_w is not None:
        plt.axhline(batch_w, linestyle='--', label="Batch Weight")

    plt.title("Weight Convergence")
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.legend()
    plt.show()


def plot_predictions(X, y, batch_model, seq_model):
    plt.figure()
    plt.scatter(X, y, alpha=0.5, label="Data")

    x_line = np.linspace(min(X), max(X), 100)

    # batch line
    w_b, b_b = batch_model
    plt.plot(x_line, w_b * x_line + b_b, label="Batch", linestyle='--')

    # sequential line
    plt.plot(x_line, seq_model.w * x_line + seq_model.b,
             label="Sequential (final)")

    plt.title("Model Comparison")
    plt.legend()
    plt.show()


# ----------------------
# Run Experiment
# ----------------------
np.random.seed(42)

N = 1000

# Toggle drift here
X, y = generate_data(N, drift=True)

# Batch
batch = Batch()
w_batch, b_batch = batch.train_model(X, y)

# Sequential models
seq_no_decay = Sequential(decay=None)
seq_decay = Sequential(decay=0.99)

# Run
errors_nd, weights_nd, _ = online_learning_system(
    seq_no_decay, data_stream(X, y)
)

errors_d, weights_d, _ = online_learning_system(
    seq_decay, data_stream(X, y)
)

# ----------------------
# Plots
# ----------------------
plot_errors({
    "No Decay": errors_nd,
    "Decay=0.99": errors_d
})

plot_weights({
    "No Decay": weights_nd,
    "Decay=0.99": weights_d
}, batch_w=w_batch)

plot_predictions(X, y, (w_batch, b_batch), seq_no_decay)