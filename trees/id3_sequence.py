# good kaggle tutorial
# https://www.kaggle.com/code/jebathuraiibarnabas/decision-tree-id3-from-scratch/notebook

import pandas as pd
import numpy as np
import math

data_path = '/home/miguel/Python_Projects/datasets/'
data = pd.read_csv(data_path + 'id3_exam.csv')
data.head()

def find_entropy(data):
    """
    Returns the entropy of the class or features
    formula: - ∑ P(X)logP(X)
    """
    entropy = 0
    for i in range(data.nunique()):
        x = data.value_counts().iloc[i]/data.shape[0] 
        entropy += (- x * math.log(x,2))
    return round(entropy,4)



def information_gain(data, data_):
    """
    Returns the information gain of the features
    """
    info = 0
    for i in range(data_.nunique()):
        df = data[data_ == data_.unique()[i]]
        w_avg = df.shape[0]/data.shape[0]
        entropy = find_entropy(df.Result)
        x = w_avg * entropy
        info += x
    ig = find_entropy(data.Result) - info
    return round(ig, 3)   



def entropy_and_infogain(datax, feature):
    """
    Grouping features with the same class and computing their 
    entropy and information gain for splitting
    """
    for i in range(data[feature].nunique()):
        df = datax[datax[feature]==data[feature].unique()[i]]
        if df.shape[0] < 1:
            continue
        
        # display(df[[feature, 'Result']].style.applymap(highlight)\
        #         .set_properties(subset=[feature, 'Result'], **{'width': '80px'})\
        #         .set_table_styles([{'selector': 'th', 'props': [('background-color', 'lightgray'), 
        #                                                         ('border', '1px solid gray'), 
        #                                                         ('font-weight', 'bold')]},
        #                            {'selector': 'td', 'props': [('border', '1px solid gray')]},
        #                            {'selector': 'tr:hover', 'props': [('background-color', 'white'), 
        #                                                               ('border', '1.5px solid black')]}]))
        
        print(f'Entropy of {feature} - {data[feature].unique()[i]} = {find_entropy(df.Result)}')
    print(f'Information Gain for {feature} = {information_gain(datax, datax[feature])}')

# Computing entropy for the entire dataset
print(f'Entropy of the entire dataset: {find_entropy(data.Result)}')

# 
entropy_and_infogain(data, 'Courses')
entropy_and_infogain(data, 'Background')
entropy_and_infogain(data, 'Working')


