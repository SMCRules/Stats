
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
More advanced than id3_class_marya.py. 
Functionality include classes. 
Attributes are continuous but discretized with thresholds
"""

class Node():
    """
    A class representing a node in a decision tree.
    """    
    def __init__(self, 
                 feature = None, 
                 threshold = None, 
                 left = None, 
                 right = None, 
                 gain = None, 
                 value = None):
        """
        Initializes a new instance of the Node class.

        Args:
            feature: The feature used for splitting at this node. Defaults to None.
            threshold: The threshold used for splitting at this node. Defaults to None.
            left: The left child node. Defaults to None.
            right: The right child node. Defaults to None.
            gain: The gain of the split. Defaults to None.
            value: If this node is a leaf node, this attribute represents the predicted value
                for the target variable. Defaults to None.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
    
class DecisionTree():
    """
    A decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class.

        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decision tree.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
        """
        Splits the given dataset into two datasets (left and right) 
        based on the given feature and threshold value.

        Parameters:
            dataset (ndarray): Input dataset.
            feature (int): Index of the feature to be split on.
            threshold (float): Threshold value to split the feature on.

        Returns:
            left_dataset (ndarray): Subset of the dataset with values less than 
            or equal to the threshold.
            right_dataset (ndarray): Subset of the dataset with values greater 
            than the threshold.
        """
        # Create empty arrays to store the left and right datasets
        left_dataset = []
        right_dataset = []
        
        # Loop over each row in the dataset and 
        # split based on the given feature and threshold
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        # Convert the left and right datasets to numpy arrays and return
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y):
        """
        Computes the entropy of the given label values.

        Parameters:
            y (ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given label values.
        """
        entropy = 0

        # Find the unique label values in y and loop over each value
        labels = np.unique(y)
        for label in labels:
            # Find the examples in y that have the current label
            label_examples = y[y == label]
            # Calculate the ratio of the current label in y
            pl = len(label_examples) / len(y)
            # Calculate the entropy using the current label and ratio
            entropy += -pl * np.log2(pl)

        # Return the final entropy value
        return entropy

    def information_gain(self, parent, left, right):
        """
        Computes the information gain (on y values)
        after splitting the parent dataset into two datasets.

        Parameters:
            parent (ndarray): Input y values for the parent dataset.
            left (ndarray): Subset of the y values (parent) after a left split.
            right (ndarray): Subset of the y values (parent) after a right split.

        Returns:
            information_gain (float): Information gain of the split.
        """
        # set initial information gain to 0
        information_gain = 0
        # compute entropy for parent
        parent_entropy = self.entropy(parent)
        # calculate weight for left and right nodes
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        # compute entropy for left and right nodes
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        # calculate weighted entropy 
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        # calculate information gain 
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    
    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best split for the given dataset
        looping through features
        looping through unique values of features (thresholds)
        compute information gain and compare with stored best gain.

        Args:
        dataset (ndarray): The dataset to split.
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split: 
        feature index, threshold, gain, left and right datasets.
        """
        # dictionary to store the best split values
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        # loop over all the features
        for feature_index in range(num_features):
            # get the feature values at feature_index
            feature_values = dataset[:, feature_index]
            # unique values of that feature
            thresholds = np.unique(feature_values)
            # loop over threshold values of the feature
            for threshold in thresholds:
                # get left and right datasets
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                # check if either datasets is empty
                if len(left_dataset) and len(right_dataset):
                    # get y values of the parent and left, right nodes
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    # compute information gain based on the y values
                    information_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if conditions are met
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split

    
    def calculate_leaf_value(self, y):
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        y = list(y)
        #get the highest present class in the array
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        # split the dataset into X, y values
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        # keeps spliting until stopping conditions are met
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # Get the best split
            best_split = self.best_split(dataset, n_samples, n_features)
            # Check that gain isn't zero. positive gain = True
            if best_split["gain"]:
                # continue splitting the left and the right child. Increment current depth
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                # return decision node
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        # compute leaf node value if not enough samples or max depth
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node value
        return Node(value=leaf_value)
    
    def fit(self, X, y):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        dataset = np.concatenate((X, y), axis=1)  
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self.make_prediction(x, self.root)
            # Append the prediction to the list of predictions
            predictions.append(prediction)
        # Convert the list to a numpy array and return it
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value 
        for the given feature vector.

        Args:
        x (ndarray): The feature vector to predict the target value for.
        node (Node): The current node being evaluated.

        Returns:
        The predicted target value for the given feature vector.
        """
        # if the node has value i.e it's a leaf node extract it's value
        if node.value != None: 
            return node.value
        else:
            #if it's node a leaf node we'll get it's feature and traverse 
            # through the tree accordingly
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)
    
    def print_tree(self, node=None, indent=""):
        """
        Prints the decision tree in a readable format with rounded threshold and gain values.

        Args:
            node (Node): The current node in the tree.
            indent (str): The indentation to show the level in the tree.
        """
        if node is None:
            node = self.root

        if node.value is not None:
            print(indent + f"Leaf: Predict -> {node.value}")
        else:
            # Round threshold to 4 decimals, gain to 3 decimals
            rounded_threshold = round(node.threshold, 4)
            rounded_gain = round(node.gain, 3)
            print(indent + f"[X{node.feature} <= {rounded_threshold}] (Gain: {rounded_gain})")
            print(indent + "  Left:")
            self.print_tree(node.left, indent + "    ")
            print(indent + "  Right:")
            self.print_tree(node.right, indent + "    ")    
        
    def tree_to_dict(self, node=None):
        """
        Converts the tree into a nested dictionary with rounded thresholds and gains for visualization.
        """
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        # Round threshold to 4 decimal places and gain to 3 decimal places
        condition = f"X{node.feature} <= {round(node.threshold, 4)} (Gain: {round(node.gain, 3)})"
        return {
            condition: {
                "True": self.tree_to_dict(node.left),
                "False": self.tree_to_dict(node.right)
            }
        }

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

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)

#create model instance
model = DecisionTree(2, 2)
# Fit the decision tree model to the training data.
model.fit(X_train, y_train)
# Print the tree structure
model.print_tree()
# Convert the tree to dictionary format
tree_dict = model.tree_to_dict()

# tree visualization
from graphviz import Digraph
import uuid

def visualize_tree(tree, parent_id=None, graph=None):
    """Recursively visualize a decision tree from a nested dictionary with rounded values."""
    if graph is None:
        graph = Digraph(format='png')
        graph.attr(size='8,8')

    for node_label, branches in tree.items():
        # Generate unique ID for each node to avoid duplicate node names
        node_id = str(uuid.uuid4())
        graph.node(node_id, label=f"{node_label}")

        if parent_id is not None:
            graph.edge(parent_id, node_id)

        if isinstance(branches, dict):
            for branch_label, subtree in branches.items():
                if isinstance(subtree, dict):
                    visualize_tree({list(subtree.keys())[0]: list(subtree.values())[0]}, node_id, graph)
                else:
                    # Create a unique leaf ID
                    leaf_id = str(uuid.uuid4())
                    graph.node(leaf_id, label=f"Predict: {subtree}", shape='box', style='filled', fillcolor='lightgray')
                    graph.edge(node_id, leaf_id, label=f"{branch_label}")

    return graph

# Generate and visualize the tree
graph = visualize_tree(tree_dict)
# graph.render("id3_fares_tree", view=True)  # Saves and opens the image

# Use the trained model to make predictions on the test data.
predictions = model.predict(X_test)

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
# Sklearn's Accuracy: 0.9469026548672567
# Sklearn's Balanced Accuracy: 0.9476027397260274

"""
plt.figure(figsize=(15, 10))
plot_tree(
        decision_tree_classifier,
        feature_names=X_cols,
        class_names=np.unique(y_train).astype(str), 
        filled=True, rounded=True
        )
# plt.savefig('sklearn_tree.png')
plt.show()
"""

