import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

def calculate_model_weight(error):
    error = np.clip(error, 1e-10, 1 - 1e-10)  # Keep error in (ε, 1-ε)
    return 0.5 * np.log((1 - error)/(error + epsilon)) 
# Update weights
def update_row_weights(row, alpha=0.423):
    if row['label'] == row['y_pred']:
        return row['weights'] * np.exp(-alpha) # Correctly classified
    else:
        return row['weights'] * np.exp(alpha)
"""
# NumPy's vectorized operations for efficiency:
df['weights'] *= np.exp(
    -alpha * (
        df['label'] == df['y_pred']
    ) + alpha * (
        df['label'] != df['y_pred']
    )
        )
"""
# create a toy dataframe
df = pd.DataFrame()
df['X1'] = [1,2,3,4,5,6,6,7,9,9]
df['X2'] = [5,3,6,8,1,9,5,8,9,2]
df['label'] = [1,1,0,1,0,1,0,1,0,0]
# visualize non-linearly separable data
sns.scatterplot(x=df['X1'],y=df['X2'],hue=df['label'])
plt.show()

#### adaboost.M1
# 1.initialize 1/N weights for every row
df['weights'] = 1/df.shape[0]
# for m = 1, ... M
# 2. train a weak classifier with max_depth = 1, using sample weights
dt1 = DecisionTreeClassifier(max_depth=1)
# Convert Pandas DataFrame to NumPy array
X = df[['X1', 'X2']].to_numpy()
y = df['label'].to_numpy()
dt1.fit(X, y, sample_weight=df['weights'])
plot_tree(dt1)
# plt.show()
plot_decision_regions(X, y, clf=dt1, legend=2)
# plt.show()
# 3. calculate error rate
df['y_pred'] = dt1.predict(X)
# 4. update weights


