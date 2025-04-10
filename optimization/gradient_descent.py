"""
The steps for the gradient descent algorithm are given below:

1. Choose a random initial point x_initial and set x[0] = x_initial

2. For iterations t=1..T
    Update x[t] = x[t-1] - eta * ∇f(x[t-1])
Gradient descent and steepest descent are the same algorithm whenever
for the direction of motion dk, is chosen as -∇f(xk). 

Let’s find the minimum of the following function of two variables:
f(x,y)=x^2+2xy+2y^2
The gradient vector is given by:
∇f(x,y) = [2x+2y,2x+4y]
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return x**2 + 2*x*y + 2*y**2

# Gradient of the function
def gradient(x, y):
    return np.array([2*x + 2*y, 2*x + 4*y])

# Gradient descent function recording the path to minimum
def gradient_descent_path(x0, y0, learning_rate=0.1, iterations=50):
    path = [(x0, y0)]
    x, y = x0, y0

    for _ in range(iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        path.append((x, y))

    return path

# Plotting function
def plot_gradient_descent_path(x0=0, y0=0, learning_rate=0.1, iterations=50):
    path = gradient_descent_path(x0, y0, learning_rate, iterations)
    x_vals, y_vals = zip(*path)

    # Create a grid of values to plot the function surface
    X, Y = np.meshgrid(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100))
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.plot(x_vals, y_vals, marker='o', color='red', label='Gradient Descent Path')
    plt.title('Gradient Descent Optimization Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.colorbar(cp, label='f(x, y)')
    plt.show()

plot_gradient_descent_path(x0 = 5, y0 = 7,learning_rate=0.1, iterations=20)