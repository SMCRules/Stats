import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor



"""
Stochastic Gradient descent applied to linear regression 
The updating sequence to update model parameters θ:
θ = θ — α ∇J(θ),

where:
θ = vector of parameters to be optimised.
α = learning rate, which sets the step size in the parameter space.
J(θ) = cost function, which measures how well the model fits the data.
∇J(θ) = gradient of the cost function with respect to θ.
"""

def sgd_linear_regression(X, y, alpha=0.01, learning_rate=0.01, n_iter=1000, tol=1e-3):
    n_samples = X.shape[0]
    
    # Add bias term (column of 1s) so weights = [w0 (bias), w1]
    X_b = np.c_[np.ones((n_samples,)), X]  # shape: (n_samples, 2)

    # Initialize weights randomly or to zero
    weights = np.zeros(X_b.shape[1])

    prev_loss = float('inf')
    
    for epoch in range(n_iter):
        for i in range(n_samples):
            xi = X_b[i]
            yi = y[i]
            
            # Prediction
            y_pred = xi @ weights
            
            # Gradient (with L2 regularization)
            error = y_pred - yi
            grad = 2 * xi * error + 2 * alpha * weights  # add regularization gradient
            
            # Update rule
            weights -= learning_rate * grad
        
        # Compute loss to check for convergence
        y_preds = X_b @ weights
        loss = np.mean((y_preds - y)**2) + alpha * np.sum(weights**2)
        
        if abs(prev_loss - loss) < tol:
            print(f"Converged after {epoch+1} epochs.")
            break
        prev_loss = loss

    return weights

### Generate synthetic data from y = -1.0 + 2.0 * X + N(0,1) error
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples)
y = -1.0 + 2.0 * X + np.random.randn(n_samples)
# 2D plot
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Reshape X and y into 2D input
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sdg scratch
weights_scratch = sgd_linear_regression(
    X_train, y_train.ravel(), alpha=0.01, learning_rate=0.01, n_iter=1000, tol=1e-3
)

print("Learned weights:", weights_scratch)

# Predict
X_test_b = np.c_[np.ones((X_test.shape[0],)), X_test]  # add bias term
y_pred_scratch = X_test_b @ weights_scratch

### sklearn implementation
# create model instance, fit data and predict on "unseen" data
# notice we flatten y to (n_samples,)
sgd = SGDRegressor(alpha=0.01, max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train.ravel())
y_pred_sklearn = sgd.predict(X_test)


mse_methods = np.mean(np.abs(y_pred_sklearn - y_pred_scratch))
print("mse between scratch and sklearn", mse_methods)
# mean squared error mean((y_test-y_pred)^2) between "unseen" y_test and 
# model predictions 
mse = mean_squared_error(y_test, y_pred_sklearn)
print(f'Test set MSE: {mse:.2f}')
# Plot the fitted line
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred_sklearn, color='red', label='SGDRegressor (sklearn)')
plt.plot(X_test, y_pred_scratch, color='blue', label='SGD from Scratch')
plt.legend(loc='upper left')
plt.xlabel('X')
plt.ylabel('y')
plt.show()