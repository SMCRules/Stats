import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


### CLASSIFICATION DATASET
df_path = "/home/miguel/Python_Projects/datasets/iris.csv"
iris = pd.read_csv(df_path)
X1, y1 = iris.iloc[:, :-1], iris.iloc[:, -1]
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=0
    )

best_alpha = 0.0143
final_tree = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0)
final_tree.fit(X1_train, y1_train)
final_tree_prediction = final_tree.predict(X1_test[:3])
print("final_tree_prediction ", final_tree_prediction)
# final_tree_prediction  ['virginica' 'versicolor' 'setosa']


