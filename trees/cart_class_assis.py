import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pprint 
from sklearn.preprocessing import LabelEncoder

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
        Calculate the Gini impurity of observations in dataset.
        
        Parameters:
        y : np.ndarray
            Target vector.
        
        Returns:
        float
            Gini impurity of the dataset.
        """
        y = y.flatten()
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
        y = y.flatten()
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
            'threshold': round(split['threshold'], 4),  # Round threshold to 4 decimal places for consistencysplit['threshold'],
            'gain': round(split['gain'], 3),
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

def data_load(file_path):
    # Identify the dataset based on the file name
    file_name = file_path.split('/')[-1]
    df = pd.read_csv(file_path)
    
    if 'breast-cancer' in file_name:        
        # Drop the redundant 'id' column
        df.drop('id', axis=1, inplace=True)
        # Encode the label into binary (0/1)
        df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
        # Compute correlations to remove weakly correlated features
        corr = df.corr()
        cor_target = abs(corr["diagnosis"])
        # Select highly correlated features (threshold = 0.25)
        relevant_features = cor_target[cor_target > 0.25].index
        df = df[relevant_features]
        X_cols = df.drop('diagnosis', axis=1).columns
        X = df.drop('diagnosis', axis=1).values
        y = df['diagnosis'].values.reshape(-1,1)
        
    elif 'diabetes' in file_name:
        X_cols = df.drop('Outcome', axis=1).columns       
        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values.reshape(-1,1)

    elif 'Iris' in file_name:        
        # Drop the 'Id' column
        df.drop('Id', axis=1, inplace=True)
        # Encode the target variable with LabelEncoder
        le = LabelEncoder()
        df['Species'] = le.fit_transform(df['Species'])
        X_cols = df.drop('Species', axis=1).columns
        X = df.drop('Species', axis=1).values
        y = df['Species'].values.reshape(-1,1)

    else:
        raise ValueError("Unsupported dataset. Please provide a valid file path.")
    
    return X, y, X_cols

### Implementation
X, y, X_cols = data_load('/home/miguel/Python_Projects/datasets/breast-cancer.xls')
print(X.shape, y.shape) 

# X, y, X_cols = data_load('/home/miguel/Python_Projects/datasets/Iris.csv')
# print(X.shape, y.shape)

#X, y, X_cols = data_load('/home/miguel/Python_Projects/datasets/diabetes.csv')
#print(X.shape, y.shape)

def scale(X):
    """
    Standardizes the data in the array X.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized features array.
    """
    # Calculate the mean and standard deviation of each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Standardize the data
    X = (X - mean) / std

    return X

X = scale(X)

def train_test_split(X, y, random_state=41, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).
        y (numpy.ndarray): Target array of shape (n_samples,).
        random_state (int): Seed for the random number generator. Default is 42.
        test_size (float): Proportion of samples to include in the test set. Default is 0.2.

    Returns:
        Tuple[numpy.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
    """
    # Get number of samples
    n_samples = X.shape[0]

    # Set the seed for the random number generator
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # Determine the size of the test set
    test_size = int(n_samples * test_size)

    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test   

def accuracy(y_true, y_pred):
    """
    Computes the accuracy of a classification model.

    Parameters:
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns:
    ----------
        float: The accuracy of the model
    """
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 

def balanced_accuracy(y_true, y_pred):
    """Calculate the balanced accuracy for a multi-class classification problem.

    Parameters
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns
    -------
        balanced_acc : The balanced accuracyof the model
        
    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    # Get the number of classes
    n_classes = len(np.unique(y_true))

    # Initialize an array to store the sensitivity and specificity for each class
    sen = []
    spec = []
    # Loop over each class
    for i in range(n_classes):
        # Create a mask for the true and predicted values for class i
        mask_true = y_true == i
        mask_pred = y_pred == i

        # Calculate the true positive, true negative, false positive, and false negative values
        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        # Calculate the sensitivity (true positive rate) and specificity (true negative rate)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        # Store the sensitivity and specificity for class i
        sen.append(sensitivity)
        spec.append(specificity)
    # Calculate the balanced accuracy as the average of the sensitivity and specificity for each class
    average_sen =  np.mean(sen)
    average_spec =  np.mean(spec)
    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

#create model instance, how about adding max_depth parameter to the model
clf = CARTClassifier()
clf.fit(X_train, y_train)

import pprint
pprint.pprint(clf.tree)

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

graph = visualize_tree(clf.tree)
# graph.render("CARTclass_assis", view=True)


# Use the trained model to make predictions on the test data.
predictions = clf.predict(X_test)
# Calculate evaluating metrics
print(f"CART Code's Accuracy: {accuracy(y_test, predictions)}")
print(f"CART Code's Balanced Accuracy: {balanced_accuracy(y_test, predictions)}")

# sklearn implementation
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Create a decision tree classifier model object.
decision_tree_classifier = DecisionTreeClassifier(criterion='gini')
# Train the decision tree classifier model using the training data.
decision_tree_classifier.fit(X_train, y_train)
# Use the trained model to make predictions on the test data.
predictions = decision_tree_classifier.predict(X_test)
# Calculate evaluating metrics
print(f"Sklearn's Accuracy: {accuracy(y_test, predictions)}")
print(f"Sklearn's Balanced Accuracy: {balanced_accuracy(y_test, predictions)}")

