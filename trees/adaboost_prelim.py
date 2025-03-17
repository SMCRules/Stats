import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions

# Define functions
def error(norm_weights, y_pred, y_true):
    """Calculate weighted error."""
    return np.dot(norm_weights, y_pred != y_true)

def calculate_model_weight(error):
    """Calculate model weight (alpha)."""
    error = np.clip(error, 1e-10, 1 - 1e-10)  # Avoid division by zero
    return 0.5 * np.log((1 - error) / error)

def update_weights(weights, y_true, y_pred, alpha):
    """Vectorized weight update function."""
    return weights * np.exp(-alpha * (y_true == y_pred) + alpha * (y_true != y_pred))

# Create a toy dataset
df = pd.DataFrame({
    'X1': [1,2,3,4,5,6,6,7,9,9],
    'X2': [5,3,6,8,1,9,5,8,9,2],
    'label': [1,1,0,1,0,1,0,1,0,0]
})

# Convert DataFrame to NumPy arrays
X = df[['X1', 'X2']].to_numpy()
y = df['label'].to_numpy()

# Visualize dataset
sns.scatterplot(x=df['X1'], y=df['X2'], hue=df['label'])
plt.show()

# Initialize weights
N = len(y)
weights = np.ones(N) / N  # Equal weights at start

# Store weak classifiers and their weights
classifiers = []
alphas = []
errors = []

# AdaBoost loop
M = 1000
for i in range(M):
    # Train weak classifier
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(X, y, sample_weight=weights)
    
    # Compute weighted error
    y_pred = dt.predict(X)
    e_i = error(weights, y_pred, y)
    errors.append(e_i)

    # Compute model weight
    alpha_i = calculate_model_weight(e_i)

    # Store classifier and weight
    classifiers.append(dt)
    alphas.append(alpha_i)

    # Update sample weights
    weights = update_weights(weights, y, y_pred, alpha_i)
    weights /= np.sum(weights)  # Normalize

# Function for final ensemble prediction
def ensemble_predict(X):
    """Compute final ensemble prediction using all classifiers and their weights."""
    weak_preds = np.array([alpha * clf.predict(X) for clf, alpha in zip(classifiers, alphas)])
    final_preds = np.sign(np.sum(weak_preds, axis=0))  # Weighted sum and sign
    return (final_preds > 0).astype(int)  # Convert to 0/1

# Create a mesh grid for visualization
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))

# Compute ensemble predictions for each point in the grid
Z = ensemble_predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot the dataset
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, edgecolor="k")

plt.title(f"Final Ensemble Decision Boundary after {M} iterations")
plt.show()

# Plot error rate over iterations
plt.plot(range(1, M+1), errors, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Weighted Error")
plt.title("Error Rate Across Iterations")
plt.show()