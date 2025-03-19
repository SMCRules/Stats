
import numpy as np

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