
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pprint 
from sklearn.preprocessing import LabelEncoder

# Classification Tree base class
class ClassificationTree:
    """
    Base class for classification trees (also known as an abstract class).
    It provides a blueprint for any specific classification tree algorithm, 
    such as ID3, C4.5, or CART. 
    The class itself does not implement the functionality, only defines a 
    common interface for all classification tree implementations.

    Definition of the Structure:
    Framework for classification trees by defining common methods:
        fit(X, y): To train the model.
        predict(X): To make predictions.
        _split(X, y): To split the dataset at a node.
        _grow_tree(X, y): To recursively build the decision tree.
    
    """

    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        raise NotImplementedError("This method should be overridden.")

    def predict(self, X):
        raise NotImplementedError("This method should be overridden.")

    def _split(self, X, y):
        raise NotImplementedError("This method should be overridden.")

    def _grow_tree(self, X, y, depth=0):
        raise NotImplementedError("This method should be overridden.")

"""
    Entropy: Is a measure of the uncertainty in the data.
    Information Gain: Calculates the information gain of a potential split by 
    comparing the entropy before and after the split.
    Best Split Selection: Iterates through all possible splits 
    (features and thresholds)to find the one that yields the highest information gain.
    Tree Growth: The _grow_tree method recursively splits the dataset 
    based on the best splits until all data points in a node belong 
    to the same class or no further splits are possible.
    Prediction: The _predict method traverses the grown tree 
    to predict the class label for a given input.
"""


class ID3Classifier(ClassificationTree):
    """
    ID3 algorithm for classification.
    """
    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for ID3Classifier class.

        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decision tree.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, max_depth=0):
        """
        Fit the ID3 classifier to the data.
        
        Parameters:
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        """
        self.tree = self._grow_tree(X, y, max_depth=0)

    def predict(self, X):
        """
        Predict the class labels for the input data.
        The list comprehension in the predict method loops through all rows 
        of the predictor dataset X and applies _predict to each row.
        
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
        # y is no longer a 1D array, we have added .reshape(-1, 1)
        # or y = y.ravel()
        y = y.flatten()  
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
        # n_ stands for number #, e_ stands for entropy 
        n_left, n_right = sum(left_indices), sum(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        # self._entropy(y) is the prior entropy before splitting 
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

        # loops through columns (features)
        # within a column loops through unique values (thresholds)
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

    def _grow_tree(self, X, y, max_depth=0):
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
        
        n_samples = X.shape[0]
        # keep spliting until stopping conditions are met
        if n_samples >= self.min_samples and max_depth <= self.max_depth:
            # Get the best split
            split = self._best_split(X, y)            
            # Check that gain isn't zero. positive gain = True   
            if split['gain']:
                left_indices = X[:, split['feature_index']] <= split['threshold']
                right_indices = X[:, split['feature_index']] > split['threshold']

                left_subtree = self._grow_tree(X[left_indices], y[left_indices], max_depth + 1)
                right_subtree = self._grow_tree(X[right_indices], y[right_indices], max_depth + 1)

                return {
                    'feature_index': split['feature_index'],
                    'threshold': round(split['threshold'], 4),  # Round threshold to 4 decimal places for consistencysplit['threshold'],
                    'gain': round(split['gain'], 3),  # Round gain to 3 decimal places for consistencysplit['gain'],
                    'left': left_subtree,
                    'right': right_subtree
                }

        return np.bincount(y).argmax()

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
clf = ID3Classifier(min_samples=2, max_depth=2)
# Fit the decision tree model to the training data.
clf.fit(X_train, y_train, max_depth=2)

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
# graph.render("id3_class_assis", view=True)

# Use the trained model to make predictions on the test data.
predictions = clf.predict(X_test)
# Calculate evaluating metrics
print(f"ID3 Code's Accuracy: {accuracy(y_test, predictions)}")
print(f"ID3 Code's Balanced Accuracy: {balanced_accuracy(y_test, predictions)}")
# Model's Accuracy: 0.9557522123893806
# Model's Balanced Accuracy: 0.9601027397260273

# sklearn implementation
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Create a decision tree classifier model object.
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
# Train the decision tree classifier model using the training data.
decision_tree_classifier.fit(X_train, y_train)
# Use the trained model to make predictions on the test data.
predictions = decision_tree_classifier.predict(X_test)
# Calculate evaluating metrics
print(f"Sklearn's Accuracy: {accuracy(y_test, predictions)}")
print(f"Sklearn's Balanced Accuracy: {balanced_accuracy(y_test, predictions)}")
