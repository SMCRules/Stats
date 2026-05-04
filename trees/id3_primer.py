"""
Code from scratch a id3 decision tree for simple binary classification
dataset. All the variables are discrete.  
The idea is to use entropy and information gain to build the tree
along with Python functions, conditions and recursions to implement
the algorithm
"""
import pandas as pd
import numpy as np
import math

def entropy(var):
    probs = var.value_counts(normalize=True)
    return -sum(probs*np.log2(probs))

# create a dataset
x1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
x2 = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
x3 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
data = pd.DataFrame([x1, x2, x3, y])





