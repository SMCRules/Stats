import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# Set seed for reproducibility
np.random.seed(42)

# Number of training and test samples
n_train, n_test = 2000, 10000
n_features = 10

# Generate standard normal features (X1, ..., X10)
X_train = np.random.randn(n_train, n_features)
X_test = np.random.randn(n_test, n_features)

# Compute sum of squared features
S_train = np.sum(X_train**2, axis=1)
S_test = np.sum(X_test**2, axis=1)

# Compute chi-square median threshold
threshold = chi2.median(df=n_features)  # 9.34 for df=10

# Assign labels based on threshold
y_train = np.where(S_train > threshold, 1, -1)
y_test = np.where(S_test > threshold, 1, -1)

# Convert to DataFrame (optional for easier handling)
train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(n_features)])
train_df['label'] = y_train

test_df = pd.DataFrame(X_test, columns=[f'X{i+1}' for i in range(n_features)])
test_df['label'] = y_test

# Plot histogram of sum of squares (for visualization)
sns.histplot(S_train, bins=30, kde=True, label="Train")
sns.histplot(S_test, bins=30, kde=True, label="Test", color='red')
plt.axvline(threshold, color='black', linestyle='dashed', label="Threshold (9.34)")
plt.legend()
plt.xlabel("Sum of Squares of Features")
plt.ylabel("Count")
plt.title("Distribution of Sum of Squared Features")
plt.show()

"""
Plan for AdaBoost Training

    Initialize weights for all training samples: wi=1/N
    For each iteration (1 to 400):
        Train a decision stump (tree with max_depth=1).
        Compute the weighted error: em=∑wi(yihm(xi)≠yi)
        Compute the model weight: αm=ln(1−em)/ln(1+em)
        Update sample weights: wi=wi*exp(-αm(yihm(xi)≠yi))
        Normalize weights: wi=wi/∑wi
    Make final predictions by combining all weak learners (weighted majority vote).
    Evaluate test error over iterations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2

# Set seed for reproducibility
np.random.seed(42)

# Number of samples and features
n_train, n_test = 2000, 10000
n_features = 10

# Generate standard normal features
X_train = np.random.randn(n_train, n_features)
X_test = np.random.randn(n_test, n_features)

# Compute sum of squared features
S_train = np.sum(X_train**2, axis=1)
S_test = np.sum(X_test**2, axis=1)

# Compute chi-square median threshold
threshold = chi2.median(df=n_features)  # 9.34 for df=10

# Assign labels: 1 if sum > threshold, otherwise -1
y_train = np.where(S_train > threshold, 1, -1)
y_test = np.where(S_test > threshold, 1, -1)

# Initialize weights for training samples
weights = np.ones(n_train) / n_train

# Store model parameters
classifiers = []
alphas = []
train_errors = []
test_errors = []

M = 600  # Number of boosting iterations

# AdaBoost training loop
for m in range(M):
    # Train a weak classifier (decision stump)
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X_train, y_train, sample_weight=weights)

    # Make predictions
    y_pred_train = stump.predict(X_train)
    y_pred_test = stump.predict(X_test)

    # Compute weighted error
    err_m = np.sum(weights * (y_pred_train != y_train)) / np.sum(weights)

    # Avoid log(0) by clipping
    err_m = np.clip(err_m, 1e-10, 1 - 1e-10)

    # Compute model weight
    alpha_m = 0.5 * np.log((1 - err_m) / err_m)

    # Update sample weights
    weights *= np.exp(-alpha_m * y_train * y_pred_train)
    weights /= np.sum(weights)  # Normalize

    # Store model
    classifiers.append(stump)
    alphas.append(alpha_m)

    # Compute ensemble predictions (weighted majority vote)
    train_ensemble_pred = np.sign(
        np.sum([a * clf.predict(X_train) for a, clf in zip(alphas, classifiers)], axis=0)
    )
    test_ensemble_pred = np.sign(
        np.sum([a * clf.predict(X_test) for a, clf in zip(alphas, classifiers)], axis=0)
    )

    # Compute and store error rates
    train_errors.append(np.mean(train_ensemble_pred != y_train))
    test_errors.append(np.mean(test_ensemble_pred != y_test))

    if m % 50 == 0:
        print(f"Iteration {m}: Train Error = {train_errors[-1]:.4f}, Test Error = {test_errors[-1]:.4f}")

# Plot test error over iterations
plt.plot(range(1, M + 1), test_errors, marker='o', linestyle='-', label="Test Error")
plt.xlabel("Boosting Iteration")
plt.ylabel("Error Rate")
plt.title(f"Test Error over {M} Iterations of AdaBoost")
plt.legend()
plt.show()