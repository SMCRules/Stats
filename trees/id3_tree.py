# https://www.kaggle.com/code/maryampirjamaat/id3-decision-tree-classifier/notebook

import pandas as pd
import numpy as np
from math import log2

def entropy(data):
    """
    input: data is the whole dataset with last column as response variable
    Calculate the entropy for the last column that contains responses yes/no.
    output: returns float entropy
    """
    labels = data.iloc[:, -1]
    unique_labels = labels.unique()
    entropy_val = 0

    for label in unique_labels:
        prob = len(labels[labels == label]) / len(labels)
        entropy_val -= prob * log2(prob)

    return entropy_val
def information_gain(data, attribute):
    """
    IG measures how much information a feature provides about a response class.
    We want the feature that decreases entropy and increases information gain.
    The function deals with one attribute or discrete predictor at a time.
    data is the whole dataset with last column the response variable yes/no.
    """
    # entropy before is the prior information entropy of responses yes/no
    entropy_before = entropy(data)
    # unique values for that specific predictor
    values = data[attribute].unique()
    # entropy after is the conditional entropy given a predictor value
    # the least the better
    entropy_after = 0

    for value in values:
        # subset splits data based on discrete value of predictor
        subset = data[data[attribute] == value]
        prob = len(subset) / len(data)        
        entropy_after += prob * entropy(subset)

    gain = entropy_before - entropy_after
    return gain
def find_best_attribute(data, attributes):
    """
    input: data is the whole dataset with last column as response variable
    output: returns the best attribute or the one with max IG to split on.
    comprehension list to loop through attributes and feed only one to IG.
    """
    gains = [information_gain(data, attr) for attr in attributes]
    best_attribute_index = np.argmax(gains)
    return attributes[best_attribute_index]

def id3(data, attributes):
    labels = data.iloc[:, -1]
    
    # if only one unique label, return that label
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    # if no attributes, return the most common label
    if len(attributes) == 0:
        return labels.mode().iloc[0]

    best_attribute = find_best_attribute(data, attributes)
    # initialize tree with dictionary
    tree = {best_attribute: {}}

    # unique values of the best attribute
    values = data[best_attribute].unique()
    for value in values:
        # Filter data where best_attribute == value
        subset = data[data[best_attribute] == value]
        # Recursive call id3 to build subtree
        # new list of attributes, excluding best_attribute. The same attribute 
        # is not used again for splitting in deeper recursive calls
        subtree = id3(subset, [attr for attr in attributes if attr != best_attribute])
        # Store the subtree in the tree dictionary
        tree[best_attribute][value] = subtree

    return tree

### implementation
file_path = "/home/miguel/Python_Projects/datasets/id3.xls"
df = pd.read_csv(file_path)
# The last column is the yes/no response variable
attributes = list(df.columns[:-1])  
tree = id3(df, attributes)

import pprint
pprint.pprint(tree)

# tree visualization
from graphviz import Digraph

def visualize_tree(tree, parent_name="", graph=None):
    """Recursively visualize an ID3 decision tree using Graphviz."""
    if graph is None:
        graph = Digraph(format='png')
        graph.attr(size='8,8')

    for key, value in tree.items():
        node_name = str(key)  # Attribute at this level
        graph.node(node_name, label=node_name)  # Create a node

        if parent_name:
            graph.edge(str(parent_name), str(node_name))  # Connect to parent

        if isinstance(value, dict):  # If there are further splits
            for attr_value, subtree in value.items():
                if isinstance(subtree, dict):  # If subtree is another split
                    visualize_tree({attr_value: subtree}, node_name, graph)
                else:  # If subtree is a leaf node
                    leaf_name = f"{attr_value} -> {subtree}"  # Use "->" instead of ":"
                    graph.node(leaf_name, label=leaf_name, shape='box', style='filled', fillcolor='lightgray')
                    graph.edge(str(node_name), str(leaf_name))  # Connect properly

    return graph

# Generate and visualize the tree
graph = visualize_tree(tree)
graph.render("decision_tree", view=True)  # Saves and opens the image

### sklearn implementation

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

def draw_decision_tree(X, y):
    # OneHotEncoder features (X)
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)])
    X_encoded = preprocessor.fit_transform(X)
    # Convert X_encoded back to a DataFrame
    X_encoded = pd.DataFrame(
        X_encoded, 
        columns=preprocessor.get_feature_names_out(X.columns)
        )

    # LabelEncoder target (y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_encoded, y_encoded)

    plt.figure(figsize=(15, 10))
    plot_tree(
        clf,
        feature_names=X_encoded.columns,
        class_names=label_encoder.classes_, 
        filled=True, rounded=True
        )
    plt.savefig('sklearn_tree.png')
    plt.show()

target_variable = 'Answer'
X = df.drop(columns=[target_variable])
y = df[target_variable]

draw_decision_tree(X, y)
