"""
Consider a FNN with one hidden layer containing two neurons for 
the make_moons dataset using off-the-shelf MLPClassifier from scikit-learn.

It is encouraged to try different hidden layer sizes and activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Create a toy dataset
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# visualize simulated dataset
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Greys, edgecolors='k', s=60)
ax.set_title(f'Make Moons Dataset')
ax.set_xlabel('Feature $x_1$')
ax.set_ylabel('Feature $x_2$')
plt.tight_layout()
plt.show()

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features on the training and apply to the test set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### neural network with one hidden layer containing two neurons. 
# The hidden layer uses the relu activation function (more stable than logistic (sigmoid)). 
# The output layer automatically uses a softmax activation function to provide probabilities when making predictions.

# Hyperparameters for the neural network
hidden_layer_depth = 1  # Number of hidden layers
hidden_layer_width = 2  # Number of neurons per hidden layer

# Create a tuple representing the hidden layer sizes
hidden_layer_sizes = (hidden_layer_width,) * hidden_layer_depth

# Train a neural network on the training dataset.
# mlp stands for multi-layer perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation='relu',  
    # activation function for hidden layers, can be 'relu' or 'sigmoid'
    solver='sgd',  # optimization algorithm
    max_iter=2000,
    learning_rate_init=0.01,
    random_state=42,
    alpha=0.01
    )

# Fitting is performed by calling the model’s fit()
mlp.fit(X_train, y_train)

# Make predictions and evaluate the testset accuracy
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# visualize training results.
# Create meshgrid for decision boundary visualization
x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]


# Get probabilities for the grid points to plot decision boundaries
probs = mlp.predict_proba(grid)[:, 1].reshape(xx.shape)


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Visualize the decision boundary
ax1 = axes[0]
ax1.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, cmap=plt.cm.Greys)
ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Greys, edgecolors='k', s=60)
ax1.set_title(f'Decision Boundary\n'
              f'Hidden Layers: {hidden_layer_depth}, Neurons per Layer: {hidden_layer_width}')
ax1.set_xlabel('Feature $x_1$')
ax1.set_ylabel('Feature $x_2$')


# Plot the training loss curve
ax2 = axes[1]
ax2.plot(mlp.loss_curve_)
ax2.set_title('Training Loss Curve')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.grid(True)

# Adjust layout for better visualization
plt.tight_layout()
plt.show()

# The decision boundary is not complex enough to separate the data
# hint: what if we increase the number of hidden layers to 3 and the number of 
# neurons to 4 above and rerun the code?
# hidden_layer_depth = 3  # Number of hidden layers
# hidden_layer_width = 4  # Number of neurons per hidden layer


import numpy as np

"""
In this example, we've created a basic neural network with one input layer 
and one output layer (thus, no hidden layers). 
The input layer has 3 neurons (as seen in the shape of X), 
and the output layer has 1 neuron (as seen in the shape of y). 
The weights are initialized randomly, and then the network is trained using 
a simple form of gradient descent, adjusting the weights based on the error 
between the predicted output and the actual output. 
The sigmoid function is used as the activation function for the neurons.
""" 

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input array
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
print("X", X)

# Output array
y = np.array([[0], [1], [1], [0]])
print("y", y)

# Seed for random number generation
np.random.seed(1)

# Initialize weights with random values
weights = 2 * np.random.random((3, 1)) - 1
print(weights)

for iter in range(10000):
    # Forward propagation
    output = sigmoid(np.dot(X, weights))

    # Compute the error
    error = y - output

    # Back propagation (using the derivative of the sigmoid function)
    adjustment = error * (output * (1 - output))    

    # Adjust the weights
    weights += np.dot(X.T, adjustment)

print("Output After Training:")
print(output)

"""
    >>> print(output)
    [[0.5]
    [0.5]
    [0.5]
    [0.5]]
    In this case our output is not making sense. 
    The weights were randomly initialized and 
    we've set the number of training iterations to 10,000. 
    However, with this basic example and the configuration of the inputs and outputs 
    (i.e., the XOR problem), it's likely the network wasn't able to correctly learn the 
    function within these constraints.

    A single layer neural network is unable to solve the XOR problem, 
    which is a non-linearly 
    separable problem, because it's essentially trying to draw a straight line 
    to separate the inputs, 
    which isn't possible. You'd need a multi-layer neural network 
    (i.e., a neural network with at least one hidden layer) to solve the XOR problem.
"""

import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output dataset
y = np.array([[0, 1, 1, 0]]).T

# Seed the random number generator
np.random.seed(1)

# Initialize weights randomly with mean 0 for the first layer (3 input nodes, 4 nodes in hidden layer)
weights0 = 2 * np.random.random((3, 4)) - 1

# Initialize weights for the second layer (4 nodes in hidden layer, 1 output node)
weights1 = 2 * np.random.random((4, 1)) - 1

for iter in range(10000):
    # Forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, weights0))
    layer2 = sigmoid(np.dot(layer1, weights1))

    # Calculate error
    layer2_error = y - layer2

    # Back propagation using the error and the derivative of the sigmoid function
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Adjust weights
    weights1 += layer1.T.dot(layer2_delta)
    weights0 += layer0.T.dot(layer1_delta)

print("Output After Training:")
print(layer2)

"""
    This is an implementation of a 2-layer neural network 
    (1 hidden layer and 1 output layer). This should correctly learn the function for the 
    XOR problem, giving you an output close to [0, 1, 1, 0]. 
    The exact values might not be 0 or 1 due to the sigmoid activation function outputting 
    values between 0 and 1. But values close to 0 can be interpreted as 0 and values close to 1 
    can be interpreted as 1.
"""
