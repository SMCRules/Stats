import numpy as np 
import pandas as pd 
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

import category_encoders as ce

## %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/home/miguel/Python_Projects/datasets/kaggle_logistic/'
test = pd.read_csv(data_path + "test.csv")
train = pd.read_csv(data_path + "train.csv")


# test_features = pd.read_csv("/home/miguel/Python_Projects/datasets/kaggle_logistic/test.csv")
# train_set = pd.read_csv("/home/miguel/Python_Projects/datasets/kaggle_logistic/train.csv")

train_target = train.target
train_features = train.drop(['target'], axis=1)
percentage = train_target.mean() * 100
print("The percentage of ones in the training target is {:.2f}%".format(percentage))
train_features.head()