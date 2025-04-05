
import numpy as np

"""
This implementation demonstrates a basic CART classifier for classification tasks. 
It constructs a decision tree based on Gini impurity and uses the tree to make 
predictions on new data.
"""

class ClassificationTree:
    """
    Base class for classification trees.
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

"""
    Gini Impurity: The _gini method calculates the Gini impurity, 
    which measures the likelihood of incorrect classification of a randomly 
    chosen element.
    Information Gain: The _information_gain method calculates the information 
    gain of a potential split based on the reduction in Gini impurity.
    Best Split: The _best_split method iterates through all possible dataset splits 
    to find the one that yields the highest information gain based on Gini impurity.
    Tree Growth: The _grow_tree method recursively splits the dataset based on 
    the best splits until all data points in a node belong to the same class 
    or no further splits are possible.
    Prediction: The _predict method traverses the grown tree to predict the class 
    label for a given input.
"""
class CARTClassifier(ClassificationTree):
    """
    CART algorithm for classification.
    """

    def fit(self, X, y):
        """
        Fit the CART classifier to the data.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        """
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the class labels for the input data.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        
        Returns:
        np.ndarray
            Predicted class labels.
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        """
        Calculate the Gini impurity of a dataset.
        
        Parameters:
        y : np.ndarray
            Target vector.
        
        Returns:
        float
            Gini impurity of the dataset.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p ** 2 for p in ps if p > 0])

    def _information_gain(self, y, left_y, right_y):
        """
        Calculate the information gain of a potential split.
        
        Parameters:
        y : np.ndarray
            Target vector.
        left_y : np.ndarray
            Left split target vector.
        right_y : np.ndarray
            Right split target vector.
        
        Returns:
        float
            Information gain of the split.
        """
        p = len(left_y) / len(y)
        return self._gini(y) - p * self._gini(left_y) - (1 - p) * self._gini(right_y)

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
        best_gain = -1
        split = {}

        for col in range(X.shape[1]):
            X_column = X[:, col]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_indices = X_column <= threshold
                right_indices = X_column > threshold
                if sum(left_indices) == 0 or sum(right_indices) == 0:
                    continue
                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    split = {
                        'feature_index': col,
                        'threshold': threshold,
                        'gain': gain
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
        if len(np.unique(y)) == 1:
            return y[0]

        if X.shape[1] == 0:
            return np.bincount(y).argmax()

        split = self._best_split(X, y)
        if split['gain'] == 0:
            return np.bincount(y).argmax()

        left_indices = X[:, split['feature_index']] <= split['threshold']
        right_indices = X[:, split['feature_index']] > split['threshold']

        left_subtree = self._grow_tree(X[left_indices], y[left_indices])
        right_subtree = self._grow_tree(X[right_indices], y[right_indices])

        return {
            'feature_index': split['feature_index'],
            'threshold': split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict(self, inputs):
        """
        Predict class for a single input.
        
        Parameters:
        inputs : np.ndarray
            Feature vector.
        
        Returns:
        int
            Predicted class label.
        """
        node = self.tree
        while isinstance(node, dict):
            if inputs[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

# Example usage
if __name__ == "__main__":
    X = np.array([[2.771244718, 1.784783929],
                  [1.728571309, 1.169761413],
                  [3.678319846, 2.81281357],
                  [3.961043357, 2.61995032],
                  [2.999208922, 2.209014212],
                  [7.497545867, 3.162953546],
                  [9.00220326, 3.339047188],
                  [7.444542326, 0.476683375],
                  [10.12493903, 3.234550982],
                  [6.642287351, 3.319983761]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    classifier = CARTClassifier()
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    print(predictions)

