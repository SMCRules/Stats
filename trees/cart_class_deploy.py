import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier, plot_tree

def tree_plot(sklearn_tree, Xa_train):
    plt.figure(figsize=(12, 18))  # customize according to the size of your tree
    plot_tree(sklearn_tree, feature_names=Xa_train.columns, filled=True, precision=6)
    plt.show()

def tree_scores_plot(estimator, ccp_alphas, train_data, test_data, metric, labels):
    train_scores, test_scores = [], []
    X_train, y_train = train_data
    X_test, y_test = test_data
    x_label, y_label = labels

    for ccp_alpha_i in ccp_alphas:
        estimator.ccp_alpha = ccp_alpha_i
        estimator.fit(X_train, y_train)
        train_pred_res = estimator.predict(X_train)
        test_pred_res = estimator.predict(X_test)

        train_score = metric(y_train, train_pred_res)
        test_score = metric(y_test, test_pred_res)
        train_scores.append(train_score)
        test_scores.append(test_score)

    print("ccp_alphas: ", ccp_alphas)
    print("test_scores: ", test_scores)
    print("train_scores: ", train_scores)
    
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label} for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()


### CLASSIFICATION DATASET
df_path = "/home/miguel/Python_Projects/datasets/iris.csv"
iris = pd.read_csv(df_path)
X1, y1 = iris.iloc[:, :-1], iris.iloc[:, -1]
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.3, random_state=0
    )

### Classification before pruning
sk_tree_classifier = DecisionTreeClassifier(random_state=0)
sk_tree_classifier.fit(X1_train, y1_train)
sk_clf_path = sk_tree_classifier.cost_complexity_pruning_path(X1_train, y1_train)
sk_clf_ccp_alphas = sk_clf_path.ccp_alphas[:-1]

sk_clf_estimator = DecisionTreeClassifier(random_state=0)
train1_data, test1_data = [X1_train, y1_train], [X1_test, y1_test]
metric = accuracy_score
labels = ['Alpha', 'Accuracy']

# tree_plot(sk_tree_classifier, X1_train)
# print(f'sklearn alphas: {sk_clf_ccp_alphas}', sep='\n')
# tree_scores_plot(sk_clf_estimator, sk_clf_ccp_alphas, train1_data, test1_data, metric, labels)

best_alpha = 0.0143
final_tree = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0)
final_tree.fit(X1_train, y1_train)
final_tree_prediction = final_tree.predict(X1_test[:3])
print("final_tree_prediction ", final_tree_prediction)
# final_tree_prediction  ['virginica' 'versicolor' 'setosa']


