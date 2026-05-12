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
        computes entropy of target labels y
        """
        probs = y.value_counts(normalize=True)
        return -sum(probs*np.log2(probs))

    def information_gain(self, xi, y):
        """
        Compute information gain for one feature xi against target y.
        xi and y should both be pandas Series.
        """
        H_outcome = self.entropy(y)
        N = len(y)
        
        weighted_entropy = (
            # pandas style avoiding looping
            y.groupby(xi)
            .apply(lambda subset_y: (len(subset_y) / N) * self.entropy(subset_y))
            .sum()
        )
        return np.round(H_outcome - weighted_entropy, 4)        

    def best_split(self, X, y):
        """
        Find the feature column in X with the highest information gain.
        """
        gains = {}
        for col in X.columns:
            gains[col] = self.information_gain(y, X[col])    
    
        return max(gains, key=gains.get)        

    def fit(self, X, y):
        """
        Train the ID3 classifier.
        """
        self.feature_names_ = list(X.columns)
        self.default_class_ = y.mode()[0]
        self.tree_ = self._build_tree(X, y)

        return self

    def _build_tree(self, X, y):
        """
        Returns a tree/subtree, not overwrite self.tree_ at every recursive call.
        Only fit() assigns the final tree to: self.tree_
        """
        
        if len(y.unique()) == 1:
            return y.iloc[0]
    
        if X.shape[1] == 0:
            return y.mode()[0]

        # Select best feature
        best = self.best_split(X, y)
        # root node of the tree with the best feature selected with IG
        tree = {best: {}}

        # Split dataset by feature values
        # For the current best feature => create one branch for each value of that feature
        for value in X[best].unique():

            mask = X[best] == value
            
            # After recursive splitting, features get removed .drop(columns=[best])
            X_subset = X.loc[mask].drop(columns=[best])
            y_subset = y.loc[mask]            

            subtree = self._build_tree(X_subset, y_subset)
            tree[best][value] = subtree

        return tree
        
    def predict_one(self, xi, tree):
        """
        Predicts a single observation, doing the recursive traversal
        Recursive internal helper, traverses the tree for ONE sample.
        """
        # Base case: leaf node
        if not isinstance(tree, dict):
            return tree

        root = next(iter(tree))    
        value = xi[root]

        # unseen category protection
        if value not in tree[root]:
            return self.default_class_

        # Traverse the tree
        subtree = tree[root][value]

        return self._predict_one(x, subtree)       
        

    def predict(self, X):
        """
        Predicts an entire dataframe, looping over rows
        High-level public API: accepts many rows 
        model.predict(X_test)   
        """
        predictions = []

        for _, row in X.iterrows():
            pred = self.predict_one(row, self.tree_)
            predictions.append(pred)            

        return predictions

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
