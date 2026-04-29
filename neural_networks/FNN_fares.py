import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# Activation functions
# =========================
def relu(Z):
    return np.maximum(0, Z), Z

def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z


# =========================
# Neural Network
# =========================
class NeuralNetwork:
    def __init__(self, layer_dimensions, learning_rate=0.01):
        self.layer_dimensions = layer_dimensions
        self.learning_rate = learning_rate
        self.parameters = {}
        self.grads = {}

    def initialize_parameters(self):
        np.random.seed(3)
        self.n_layers = len(self.layer_dimensions)

        for l in range(1, self.n_layers):
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_dimensions[l],
                self.layer_dimensions[l - 1]
            ) * np.sqrt(2 / self.layer_dimensions[l - 1])

            self.parameters[f'b{l}'] = np.ones((self.layer_dimensions[l], 1)) * 0.01

    def forward_propagation(self, X):        
        caches = []
        A = X
        L = self.n_layers - 1

        for l in range(1, L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A_prev) + b
            A, activation_cache = relu(Z)
            
            caches.append((A_prev, W, b, activation_cache))

        # Output layer
        A_prev = A
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']

        Z = np.dot(W, A_prev) + b
        AL, activation_cache = sigmoid(Z)
        caches.append((A_prev, W, b, activation_cache))

        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        AL = np.clip(AL, 1e-9, 1 - 1e-9)

        cost = -1/m * np.sum(
            Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
        )
        return cost

    def backward_propagation(self, AL, Y, caches):
        m = Y.shape[1]
        L = self.n_layers - 1

        Y = Y.reshape(AL.shape)

        # Output layer (simplified!)
        A_prev, W, b, Z = caches[-1]
        dZ = AL - Y
        self.grads[f'dW{L}'] = (1/m) * np.dot(dZ, A_prev.T)
        self.grads[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        # Hidden layers
        for l in reversed(range(1, L)):
            A_prev, W, b, Z = caches[l-1]

            dZ = relu_backward(dA_prev, Z)
            self.grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
            self.grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

    def update_parameters(self):
        for l in range(1, self.n_layers):
            # 🔍 Sanity check
            # print(f"Layer {l}: W{l} {self.parameters[f'W{l}'].shape}, dW{l} {self.grads[f'dW{l}'].shape}")

            self.parameters[f'W{l}'] -= self.learning_rate * self.grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.grads[f'db{l}']    

    def predict(self, X):
        X = X.T
        AL, _ = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions.flatten()

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def create_mini_batches(self,X, Y, batch_size):
        m = X.shape[1]
        permutation = np.random.permutation(m)
        
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        mini_batches = []
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]
            mini_batches.append((X_batch, Y_batch))
            
        return mini_batches

    def fit(self, X, Y, epochs=1000, batch_size=64, print_cost=True):
        X = X.T
        Y = Y.reshape(1, -1)

        self.initialize_parameters()

        for i in range(epochs):
            mini_batches = self.create_mini_batches(X, Y, batch_size)

            for X_batch, Y_batch in mini_batches:
                AL, caches = self.forward_propagation(X_batch)
                self.backward_propagation(AL, Y_batch, caches)
                self.update_parameters()

            
            # Compute cost on full dataset (optional)
            if print_cost and i % 500 == 0:
                AL_full, _ = self.forward_propagation(X)
                cost = self.compute_cost(AL_full, Y)
                print(f"Cost at epoch {i}: {cost:.4f}")

        # for i in range(epochs):
        #     AL, caches = self.forward_propagation(X)
        #     cost = self.compute_cost(AL, Y)
        #     self.backward_propagation(AL, Y, caches)
        #     self.update_parameters()

        #     if print_cost and i % 1000 == 0:
        #         print(f"Cost at iteration {i}: {cost:.4f}")
    
# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # from sklearn.datasets import load_breast_cancer
    # data = load_breast_cancer()
    # X, y = data.data, data.target

    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    layers = [2, 16, 16, 1]
    model = NeuralNetwork(layer_dimensions=layers, learning_rate=0.01)
    model.fit(X, y, epochs=5000, batch_size=64)
    
    # Model
    # layers = [X.shape[1], 32, 16, 1]
    # model = NeuralNetwork(layer_dimensions=layers, learning_rate=0.01)
    # model.fit(X_train, y_train, epochs=10000)

    acc = model.accuracy(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f}")

    def plot_decision_boundary(model, X, y):
        import matplotlib.pyplot as plt
        
        # Only works for 2D features
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        h = 0.01  # grid step
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title("Decision Boundary")
        plt.show()

    # Plot boundary
    plot_decision_boundary(model, X, y)

