import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pprint 
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_linnerud

"""
    These optimizations can help make the CART regressor more efficient and 
    robust, especially when working with large datasets.

   * Maximum Depth (max_depth): Limits the depth of the tree to prevent it 
    from growing too complex. This helps in reducing overfitting and 
    improves generalization.
   * Minimum Samples Split (min_samples_split): Ensures that a node must have 
    at least a certain number of samples before it can be split. 
    This prevents the model from creating overly specific branches that 
    may not generalize well.
   * Efficient Split Evaluation: The _best_split method is optimized 
    to handle only feasible splits by checking the minimum number of 
    samples required for a split.
"""

import numpy as np
from sklearn.utils import resample

class RegressionTree:
    """
    Base class for regression trees.
    """

    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        raise NotImplementedError("This method should be overridden.")

    def predict(self, X):
        raise NotImplementedError("This method should be overridden.")

    def _split(self, X, y):
        raise NotImplementedError("This method should be overridden.")

    def _grow_tree(self, X, y):
        raise NotImplementedError("This method should be overridden.")

class CARTRegressor(RegressionTree):
    """
    Optimized CART algorithm for regression with complexity pruning.
    """

    def __init__(self, max_depth=None, min_samples_split=20):
        """
        Initialize the CART regressor.
        
        Parameters:
        max_depth : int or None
            Maximum depth of the tree.
        min_samples_split : int
            Minimum number of samples required to split an internal node.
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        

    def fit(self, X, y):
        """
        Fit the CART regressor to the data.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        """
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        
        Returns:
        np.ndarray
            Predicted target values.
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _mean_squared_error(self, y_true, y_pred):
        """
        Calculate the mean squared error between true and predicted values.
        
        Parameters:
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        
        Returns:
        float
            Mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _best_split(self, X, y):
        """
        Find the best split for a dataset.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        
        Returns:
        dict
            Best split information.
        """
        # bad mse
        best_mse = float('inf')
        # always return a complete dictionary, even if no valid split is found
        # avoid an empty dictionary split = {}
        best_split = {
            'feature_index': None,
            'threshold': None,
            'mse': float('inf')
        }

        num_features = X.shape[1]
        for col in range(num_features):
            X_column = X[:, col]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_indices = X_column <= threshold
                right_indices = X_column > threshold

                if sum(left_indices) < self.min_samples_split or sum(right_indices) < self.min_samples_split:
                    continue

                left_y, right_y = y[left_indices], y[right_indices]
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)
                mse = (
                    len(left_y) * self._mean_squared_error(left_y, left_mean) +
                    len(right_y) * self._mean_squared_error(right_y, right_mean)
                ) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature_index': col,
                        'threshold': threshold,
                        'mse': mse
                    }

        return best_split

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree with pruning.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        depth : int
            Current depth of the tree.
        
        Returns:
        dict or float
            Grown decision tree or target value.
        """
        if len(np.unique(y)) == 1:
            return y[0]

        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y)

        if len(y) < self.min_samples_split:
            return np.mean(y)        

        split = self._best_split(X, y)
        if split['mse'] == float('inf'):
            return np.mean(y)

        left_indices = X[:, split['feature_index']] <= split['threshold']
        right_indices = X[:, split['feature_index']] > split['threshold']

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature_index': split['feature_index'],
            'threshold': round(split['threshold'], 4),
            'mse': round(split['mse'], 3),
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict(self, inputs):
        """
        Predict target value for a single input.
        
        Parameters:
        inputs : np.ndarray
            Feature vector.
        
        Returns:
        float
            Predicted target value.
        """
        node = self.tree
        while isinstance(node, dict):
            if inputs[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

X, y = load_linnerud(return_X_y=True, as_frame=True)
y = y['Pulse']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()    
y_test = y_test.to_numpy()

# print(X_train, y_train, sep='\n')
rgr = CARTRegressor(max_depth=3, min_samples_split=2)
rgr.fit(X_train, y_train)

import pprint
pprint.pprint(rgr.tree)

# tree visualization
from graphviz import Digraph
import uuid

def visualize_tree(tree, parent_id=None, graph=None):
        
    if graph is None:
        graph = Digraph(format='png')
        graph.attr(size='8,8')

    # If it's a leaf node
    if not isinstance(tree, dict):
        node_id = str(uuid.uuid4())
        graph.node(node_id, label=f"Predict: {tree}", shape='box', style='filled', fillcolor='lightgray')
        if parent_id is not None:
            graph.edge(parent_id, node_id)
        return graph

    # Build node label with feature, threshold, and gain
    feature = tree['feature_index']
    threshold = round(tree['threshold'], 4)
    gain = round(tree.get('gain', 0), 3)  # use .get in case gain is missing

    node_label = f"X{feature} <= {threshold} (Gain: {gain})"
    node_id = str(uuid.uuid4())
    graph.node(node_id, label=node_label)

    if parent_id is not None:
        graph.edge(parent_id, node_id)

    # Recurse for left and right subtrees
    visualize_tree(tree['left'], parent_id=node_id, graph=graph)
    visualize_tree(tree['right'], parent_id=node_id, graph=graph)

    return graph

graph = visualize_tree(rgr.tree)
graph.render("CART_regres_pruning_assis", view=True)
rgr_preds = rgr.predict(X_test)


### sklearn implementation
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error

sk_tree_regressor = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3, min_samples_split=2, 
    random_state=0
    )
sk_tree_regressor.fit(X_train, y_train)
sk_preds = sk_tree_regressor.predict(X_test)
sk_error = mean_absolute_percentage_error(y_test, sk_preds)
print("sk_error: ", sk_error)
rgr_error = mean_absolute_percentage_error(y_test, rgr_preds)
print("rgr_error: ", rgr_error)