
import numpy as np

# Classification Tree base class
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
        raise NotI

class ID3Classifier(ClassificationTree):
    """
    ID3 algorithm for classification.
    """

    def fit(self, X, y):
        """
        Fit the ID3 classifier to the data.
        
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

    def _entropy(self, y):
        """
        Calculate the entropy of a dataset.
        
        Parameters:
        y : np.ndarray
            Target vector.
        
        Returns:
        float
            Entropy of the dataset.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _information_gain(self, X_column, y, split_threshold):
        """
        Calculate the information gain of a potential split.
        
        Parameters:
        X_column : np.ndarray
            Feature column.
        y : np.ndarray
            Target vector.
        split_threshold : float
            Value to split the feature on.
        
        Returns:
        float
            Information gain of the split.
        """
        left_indices = X_column <= split_threshold
        right_indices = X_column > split_threshold
        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = sum(left_indices), sum(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        return self._entropy(y) - child_entropy

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
                gain = self._information_gain(X_column, y, threshold)
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

    classifier = ID3Classifier()
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    print(predictions)