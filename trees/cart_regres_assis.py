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
This implementation provides a basic CART regressor that builds a decision tree 
to predict continuous target values based on the mean squared error criterion.
"""

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
    CART algorithm for regression.
    """

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
        # start with a bad mse
        best_mse = float('inf')
        split = {}

        for col in range(X.shape[1]):
            X_column = X[:, col]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_indices = X_column <= threshold
                right_indices = X_column > threshold
                # check whether threshold leave left and right predictor values
                if sum(left_indices) == 0 or sum(right_indices) == 0:
                    continue
                # assign observations to left and right 
                left_y, right_y = y[left_indices], y[right_indices]
                # sample means are predictions for groups
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)
                mse = (len(left_y) * self._mean_squared_error(left_y, left_mean) +
                       len(right_y) * self._mean_squared_error(right_y, right_mean)) / len(y)
                # if mse is improved, consider the split
                if mse < best_mse:
                    best_mse = mse
                    split = {
                        'feature_index': col,
                        'threshold': threshold,
                        'mse': mse
                    }

        return split

    def _grow_tree(self, X, y):
        """
        Recursively grow the decision tree.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        
        Returns:
        dict
            Grown decision tree.
        """
        # Stopping conditions: a unique dependent value
        if len(np.unique(y)) == 1:
            return y[0]
        # no predictors left
        if X.shape[1] == 0:
            return np.mean(y)

        split = self._best_split(X, y)
        if split['mse'] == float('inf'):
            return np.mean(y)

        left_indices = X[:, split['feature_index']] <= split['threshold']
        right_indices = X[:, split['feature_index']] > split['threshold']

        left_subtree = self._grow_tree(X[left_indices], y[left_indices])
        right_subtree = self._grow_tree(X[right_indices], y[right_indices])

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
rgr = CARTRegressor()
rgr.fit(X_train, y_train)

import pprint
pprint.pprint(rgr.tree)

# tree visualization
from graphviz import Digraph
import uuid

def visualize_tree(tree, parent_id=None, graph=None):
    """Visualize a decision tree from nested dict with feature_index, threshold, gain, left, right."""
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
# graph.render("CART_regres_assis", view=True)

rgr_preds = rgr.predict(X_test)

### sklearn implementation
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error

sk_tree_regressor = DecisionTreeRegressor(random_state=0)
sk_tree_regressor.fit(X_train, y_train)
sk_preds = sk_tree_regressor.predict(X_test)
sk_error = mean_absolute_percentage_error(y_test, sk_preds)
print("sk_error: ", sk_error)

rgr_error = mean_absolute_percentage_error(y_test, rgr_preds)
print("rgr_error: ", rgr_error)