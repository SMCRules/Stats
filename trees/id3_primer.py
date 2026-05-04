"""
Code from scratch a id3 decision tree for simple binary classification
dataset. All the variables are discrete.  
The idea is to use entropy and information gain to build the tree
along with Python functions, conditions and recursions to implement
the algorithm
"""
import pandas as pd
import numpy as np
import math

def entropy(var):
    probs = var.value_counts(normalize=True)
    return -sum(probs*np.log2(probs))

# def information_gain(outcome, feature):
#     """
#     Your implementation assumes binary features (0 and 1):
#     But it will break or give wrong results if:
#     feature has more than 2 categories
#     feature is not encoded as 0/1
#     """
#     # H(parent) = entropy outcome
#     Nobs = len(outcome)
#     H_outcome = entropy(outcome)
#     # split outcome by feature
#     split_left = outcome[feature == 1]
#     split_right = outcome[feature == 0]
#     #
#     Nleft = len(split_left)
#     Nright = len(split_right)
#     #
#     H_left = entropy(split_left)
#     H_right = entropy(split_right)
#     ig = H_outcome - ((Nleft/Nobs*H_left) + (Nright/Nobs*H_right))
    
#     return np.round(ig, 4)

def information_gain(outcome, feature):
    """
    General Information Gain calculation for:
    binary features
    multi-class categorical features
    But:
    For continuous features, we will need thresholds (like in CART)
    """
    H_outcome = entropy(outcome)
    N = len(outcome)
    # weighted_entropy = 0
    # for value in feature.unique():
    #     subset = outcome[feature == value]        
    #     weight = len(subset) / N
    #     weighted_entropy += weight * entropy(subset)
    
    # pandas style avoiding looping
    weighted_entropy = (
        outcome.groupby(feature)
        .apply(lambda x: (len(x)/N) * entropy(x))
        .sum()
    )            

    return np.round(H_outcome - weighted_entropy, 4)

def best_feature(X, y):
    """
    Chooses the feature with highest IG
    So, this becomes the internal node
    """
    gains = {}
    for col in X.columns:
        gains[col] = information_gain(y, X[col])
        # print(col, gains[col])
    
    return max(gains, key=gains.get)

def id3(X, y):
    # Stopping condition 1: pure node
    if len(y.unique()) == 1:
        return y.iloc[0]
    
    # Stopping condition 2: no features left
    if X.shape[1] == 0:
        return y.mode()[0]

    # Select best feature
    best = best_feature(X, y)
    tree = {best: {}}

    # Split dataset by feature values
    for value in X[best].unique():
        X_subset = X[X[best] == value].drop(columns=[best])
        y_subset = y[X[best] == value]

        subtree = id3(X_subset, y_subset)
        tree[best][value] = subtree

    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    root = next(iter(tree))
    value = sample[root]
    subtree = tree[root][value]

    return predict(subtree, sample)

# create a dataset
x1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
x2 = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
x3 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
data = pd.DataFrame({
    'X1': x1,
    'X2': x2,
    'X3': x3,
    'Y': y
})

X = data.drop('Y', axis=1)
y = data.Y
for i in range(data.shape[1]):
    print(f'Entropy of {data.columns[i]} = {entropy(data[data.columns[i]])}')
    print(f'IG for {data.columns[i]} = {information_gain(data.Y, data[data.columns[i]])}')
    print(f'Best feature for {data.columns[i]} = {best_feature(X, data.Y)}')

tree = id3(X, y)
print(tree)

data_path = '/home/miguel/Python_Projects/datasets/'
data = pd.read_csv(data_path + 'PlayTennis.csv')
print(data.head())
X = data.drop('play', axis=1)
y = data['play']

tree = id3(X, y)
print(tree)
