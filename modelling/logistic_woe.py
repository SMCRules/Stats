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
# print("The percentage of ones in the training target is {:.2f}%".format(percentage))
# print(train_features.head(5))
# print(train_target.head(5))

# Example: encoding train_features with WOE and appending to raw features  
columns = [col for col in train_features.columns if col != 'id']

woe_encoder = ce.WOEEncoder(cols=columns)
woe_encoded_train = woe_encoder.fit_transform(
    train_features[columns], train_target
    ).add_suffix('_woe')
# joining woe encoded features to raw features
train_features = train_features.join(woe_encoded_train)
woe_encoded_cols = woe_encoded_train.columns

# Let's check the effect of woe enconding on feature nom_0.
"""
WoE is a measure commonly used in binary classification tasks 
to quantify the predictive power of a categorical variable by comparing 
the distribution of good (0) vs. bad (1) outcomes across its categories.

WoE gives a log-odds measure of how a category predicts the target.
Small WoE values (close to 0) mean little predictive power.
Positive WoE → higher probability of target = 1
Negative WoE → higher probability of target = 0
"""
df_appended = train_features.copy()
df_appended['target'] = train_target

overall_number_of_ones = train_target.sum()
overall_number_of_zeroes = train_target.shape[0] - overall_number_of_ones

print("There are {} ones and {} zeroes in the training set".format(
    overall_number_of_ones, overall_number_of_zeroes
))

grouped = pd.DataFrame()
grouped['Total'] = df_appended.groupby('nom_0').id.count()
grouped['number of ones'] = df_appended.groupby('nom_0').target.sum()
grouped['number of zeroes'] = grouped['Total'] - grouped['number of ones']

grouped['percentage of ones'] = grouped['number of ones'] / overall_number_of_ones
grouped['percentage of zeroes'] = grouped['number of zeroes'] / overall_number_of_zeroes
grouped['(% ones) > (% zeroes)'] = grouped['percentage of ones'] > grouped['percentage of zeroes']

grouped['weight of evidence'] = df_appended.groupby('nom_0').nom_0_woe.mean()
print(grouped)


# Performance comparison with other encoders

# helper function
def logreg_test(cols, encoder):
    df = train_features[cols]
    auc_scores = []
    acc_scores = []
    
    skf = StratifiedKFold(n_splits=6, shuffle=True).split(df, train_target)
    for train_id, valid_id in skf:
        enc_tr = encoder.fit_transform(df.iloc[train_id,:], train_target.iloc[train_id])
        enc_val = encoder.transform(df.iloc[valid_id,:])
        regressor = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.6)
        regressor.fit(enc_tr, train_target.iloc[train_id])
        acc_scores.append(regressor.score(enc_val, train_target.iloc[valid_id]))
        probabilities = [pair[1] for pair in regressor.predict_proba(enc_val)]
        auc_scores.append(roc_auc_score(train_target.iloc[valid_id], probabilities))
        
    acc_scores = pd.Series(acc_scores)
    mean_acc = acc_scores.mean() * 100
    print("Mean accuracy score: {:.3f}%".format(mean_acc))
    
    auc_scores = pd.Series(auc_scores)
    mean_auc = auc_scores.mean() * 100
    print("Mean AUC score: {:.3f}%".format(mean_auc))

##########################################
print("Using Weight of Evidence Encoder")
woe_encoder = ce.WOEEncoder(cols=columns)
logreg_test(columns, woe_encoder)

##########################################
print("\nUsing Target Encoder")
targ_encoder = ce.TargetEncoder(cols=columns, smoothing=0.2)
logreg_test(columns, targ_encoder)

##########################################
print("\nUsing CatBoost Encoder")
cb_encoder = ce.CatBoostEncoder(cols=columns)
logreg_test(columns, cb_encoder)





