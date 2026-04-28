# https://www.kaggle.com/code/salmayassin/catboost-tutorial/notebook
# practical implementation
import catboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import time

#Loading data
data_path = '/home/miguel/Python_Projects/datasets/'
adult = pd.read_excel(data_path + 'adult.xlsx')
adult.shape
adult.isna().sum()

# income to binary variable
adult['Income']=adult['Income'].replace('<=50K',0,regex=True)
adult['Income']=adult['Income'].replace('>50K',1,regex=True)
adult.head()


# to split categorical from numeric features
df_numeric=pd.DataFrame()
for variable in adult.columns:
      if adult[variable].dtype!='object'and variable!='Income':
        df_numeric=pd.concat([df_numeric, adult[variable]],axis=1)

df_cat=pd.DataFrame()
for variable in adult.columns:
      if adult[variable].dtype=='object' :
        df_cat=pd.concat([df_cat, adult[variable]],axis=1)

cat=df_cat.to_numpy()
Full_data=np.concatenate((df_numeric,cat),axis=1)
y=adult['Income'].to_numpy()
X=Full_data

X.shape

# splitting data into train,val , test
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

#function to evaluate model performance
def evaluate_model (y_test,y_pred):
      print(f'Accuracy = {round(accuracy_score(y_test,y_pred)*100,1)} %')
      print(f'Precision = {round(precision_score(y_test,y_pred)*100,1)} %')
      print(f'Recall = {round(recall_score(y_test,y_pred)*100,1)}%')
      print(f'F1 score = {round(f1_score(y_test,y_pred)*100,1)}%')

#automaticly set learning rate  based on Logloss, MultiClass & RMSE loss functions depending on the number of iterations
#if none of parameters leaf_estimation_iterations, --leaf-estimation-method,l2_leaf_reg is set.
clf=catboost.CatBoostClassifier(iterations=1000,verbose=100,random_seed=42)
start=time.time()
clf.fit(X_train,y_train,cat_features=[6,7,8,9,10,11,12,13],eval_set=(X_val,y_val))
end=time.time()
training_time=end-start
print(f'Model fitted in {round(training_time,2)} seconds')

y_pred=clf.predict(X_test)
evaluate_model(y_test,y_pred)

#learning rate=0.01
#default number of trees(iterations)=1000, verbose =100 , to show results every 100 iterations
clf=catboost.CatBoostClassifier(iterations=1000,learning_rate=0.01,verbose=100,random_seed=42)
start=time.time()
clf.fit(X_train,y_train,cat_features=[6,7,8,9,10,11,12,13],eval_set=(X_val,y_val))
end=time.time()
training_time=end-start
print(f'Model fitted in {round(training_time,2)} seconds')

y_pred=clf.predict(X_test)
evaluate_model(y_test,y_pred)

#learning rate=0.1
clf=catboost.CatBoostClassifier(learning_rate=0.1,verbose=100,random_seed=42)
start=time.time()
clf.fit(X_train,y_train,cat_features=[6,7,8,9,10,11,12,13],eval_set=(X_val,y_val))
end=time.time()
training_time=end-start
print(f'Model fitted in {round(training_time,2)} seconds')

y_pred=clf.predict(X_test)
evaluate_model(y_test,y_pred)

# 2) L2 leaf regularization : is used to smooth weights to avoid overfitting