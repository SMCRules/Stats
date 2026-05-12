"""
ID3 code with OOP:
What objects exist, what data do they own, and what behaviour belongs to them?

Procedural functions become methods of an ID3Classifier object:
1. Entropy 
2. Information Gain
3. Best Feature Selection
3. ID3 Algorithm for building tree
4. Prediction
5. Plotting the tree
"""
import pandas as pd
import numpy as np
import math

class Node:
    def __init__(self, feature=None, prediction=None):
        self.feature = feature
        self.prediction = prediction
        self.children = {}

    def is_leaf(self):
        return self.prediction is not None

class ID3Classifier:
    def __init__(self):
        self.tree_ = None
        self.feature_names_ = None
        self.default_class_ = None

    def entropy(self, y):
        """
        computes entropy on target labels
        """
        probs = y.value_counts(normalize=True)
        return -sum(probs*np.log2(probs))

    def information_gain(self, xi, y):
        """
        Computes the IG for a given feature xi.        
        """
        H_outcome = entropy(y)
        N = len(y)
        
        weighted_entropy = (
            # pandas style avoiding looping
            y.groupby(xi)
            .apply(lambda x: (len(x)/N) * entropy(x))
            .sum()
        )
        return np.round(H_outcome - weighted_entropy, 4)

        

    def best_split(self, X, y, features):
        ...

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, features)
        return self

    def _build_tree(self, X, y, features):
        ...

    def predict_one(self, x):
        ...

    def predict(self, X):
        ...

def entropy(input):
    probs = input.value_counts(normalize=True)
    # print("probs", probs)
    return -sum(probs*np.log2(probs))

# def information_gain(outcome, feature):
#     """
#     This implementation assumes binary features (0 and 1):
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
    Computes the IG for a given feature.
    Information Gain: how much uncertainty about outcome is reduced 
    after a split by feature. 
    We are dealing with categorical features that naturally split outcome
    For continuous features, we will need thresholds (like in CART)
    """
    H_outcome = entropy(outcome)
    N = len(outcome)
    # weighted_entropy = 0
    # for value in feature.unique():
    #     subset = outcome[feature == value]        
    #     weight = len(subset) / N
    #     weighted_entropy += weight * entropy(subset)    
    
    # outcome.groupby(feature) = groups the target by feature values
    # apply(lambda x: (len(x)/N) * entropy(x)) = apply function to each group
    # of features for each subset x
    weighted_entropy = (
        # pandas style avoiding looping
        outcome.groupby(feature)
        .apply(lambda x: (len(x)/N) * entropy(x))
        .sum()
    )            

    return np.round(H_outcome - weighted_entropy, 4)

def best_feature(X, y):
    """
    Loop through all the features, compute IG for each and 
    form a gains dictionary with feature name as key and IG as value. 
    max(iterable, key=function): Apply function to every element and 
    maximize according to the returned values. gains.get('x1') → 1.0     
    """
    gains = {}
    for col in X.columns:
        gains[col] = information_gain(y, X[col])
        # print(col, gains[col])
    
    # the returned value is the key (feature) 
    # associated with the highest IG
    return max(gains, key=gains.get)

def id3_tree(X, y):
    # Do all remaining observations belong to the same class?
    # Stopping condition 1: return a leaf node: final prediction
    # once all labels (target) are identical no more IG is possible
    if len(y.unique()) == 1:
        return y.iloc[0]
    
    # no more attributes available to separate classes, so predict the majority class
    # Stopping condition 2: no features left
    if X.shape[1] == 0:
        return y.mode()[0]

    # Select best feature
    best = best_feature(X, y)
    # root node of the tree with the best feature selected with IG
    tree = {best: {}}

    # Split dataset by feature values
    # For the current best feature => create one branch for each value of that feature
    for value in X[best].unique():
        print("best, value\n", best, value)
        # After recursive splitting, features get removed .drop(columns=[best])
        X_subset = X[X[best] == value].drop(columns=[best])
        print("X_subset\n", X_subset)
        y_subset = y[X[best] == value]
        print("y_subset\n", y_subset)

        subtree = id3_tree(X_subset, y_subset)
        tree[best][value] = subtree

    return tree

def predict(tree, sample):
    """
    The prediction function mirrors the recursive structure of the ID3 tree,
    walking down the tree following sample feature values until reaching a leaf.
    """
    # Stopping condition
    # Tests whether we have reached a leaf node 'not dictionary' 
    # instead of another subtree 'dictionary'
    # isinstance(object, type): checks whether an object belongs to a given class/type
    # isinstance('yes', dict) → False returns a prediction
    if not isinstance(tree, dict):
        return tree
    # iter(tree) = creates an iterator over the dictionary keys. 
    # next() extracts the next element from an iterator.
    root = next(iter(tree))    
    value = sample[root]
    # Traverse the tree
    subtree = tree[root][value]

    # Recursive prediction: calls prediction again on the smaller subtree.
    return predict(subtree, sample)

def data_load(file_path=None, synthetic=False):
    # Synthetic data
    if synthetic:
        x1 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        x2 = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        x3 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y =  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        df = pd.DataFrame({
            'X1': x1,
            'X2': x2,
            'X3': x3,
            'Y': y
        })

        target = 'Y'
    
    else:
        # Identify the dataset based on the file name
        file_name = file_path.split('/')[-1]
        df = pd.read_csv(file_path)
        
        if 'exam' in file_name:
            target = 'Working'
                        
        elif 'tennis' in file_name:
            target = 'PlayTennis'
                    
        else:
            raise ValueError("Unsupported dataset. Please provide a valid file path.")

    X = df.drop(columns=[target])
    y = df[target]
    X_cols = X.columns   
    
    return X, y, X_cols

### Implementation pick either option
#X, y, X_cols = data_load('/home/miguel/Python_Projects/datasets/id3_tennis.csv')
#X, y, X_cols = data_load('/home/miguel/Python_Projects/datasets/id3_exam.csv')
X, y, X_cols = data_load(synthetic=True)

data = X.copy()
data['y'] = y
print(X.shape, y.shape)
print("all data\n", data)

tree = id3_tree(X, y)
print(tree)

"""
# important pattern in ML
gains = {
    'x1': 1.0,
    'x2': 0.029,
    'x3': 0.029
}
max(
    ['x1', 'x2', 'x3'],
    key=lambda k: gains[k]
)
best_model = max(models, key=models.get)
best_split = max(candidate_splits, key=lambda s: s['gain'])
When iterating over a dictionary, Python iterates over the keys, not the values.
"""
