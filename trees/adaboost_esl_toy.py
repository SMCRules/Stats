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