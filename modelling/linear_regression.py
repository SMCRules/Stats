"""
the Ames Housing dataset, contains information about individual residential property 
in Ames, Iowa, from 2006 to 2010. The dataset was collected by Dean De Cock in 2011, 
and additional information is available via the following links:

• A report describing the dataset: http://jse.amstat.org/v19n3/decock.pdf
• Detailed documentation regarding the dataset's features: 
http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

• The dataset in a tab-separated format is available at: 
http://jse.amstat.org/v19n3/decock/AmesHousing.txt

### dataset description
The Ames Housing dataset consists of 2,930 examples and 80 features. 
For simplicity, we will only work with a subset of the features, 
shown in the following list and selected with columns: 
•Overall Qual: Rating for the overall material and finish of the house 
on a scale from 1 (very poor) to 10 (excellent)
•Overall Cond: Rating for the overall condition of the house 
on a scale from 1 (very poor) to 10 (excellent)
•Gr Liv Area: Above grade (ground) living area in square feet
•Central Air: Central air conditioning (N=no, Y=yes)
•Total Bsmt SF: Total square feet of the basement area

We will consider sale price (SalePrice) as our target variable or the dependent
variable that we want to model and predict:
•SalePrice: Sale price in U.S. dollars ($)

Thus, we have five explanatory variables and one target variable. 
"""
# import relevant python packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

columns = [
    'Overall Qual', 'Overall Cond', 'Gr Liv Area', 
    'Central Air', 'Total Bsmt SF', 'SalePrice'
    ]
data_path = 'http://jse.amstat.org/v19n3/decock/AmesHousing.txt'
df = pd.read_csv(data_path, delimiter = '\t', usecols=columns)
# display the first 5 rows of the predictors
print(df.head(5))
# data frame dimensions
print(df.shape)
# check the types of the columns
print(df.dtypes)
# Central Air is categorical but encoded as a string
# convert 'Y' to integer 1, and 'N' to the integer 0
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

# check the number of missing values in each column
print(df.isnull().sum())
# Total Bsmt SF contains one missing value. We have 2930 observations or rows 
# in the dataset. The easiest way to deal with a missing value is to remove 
# the corresponding row from the dataset.

# drop rows with missing values
df = df.dropna(axis=0)
print(df.isnull().sum())

### Exploratory Data Analysis
# scatterplot matrix provides a graphical summary of relationships in a dataset
sns.pairplot(df, corner=True, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle("Scatterplot Matrix", y=1.02)
plt.tight_layout()
plt.show()

# Notice bottom row, shows SalePrice against each predictor
# appreciate linearish relationship between SalePrice-Gr Liv Area,
# SalePrice-Total Bsmt SF and SalePrice-Overall Qual.

# Relationships using correlation matrix
corrmat = np.corrcoef(df.values.T)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corrmat, 
    xticklabels=columns, 
    yticklabels=columns, 
    annot=True, 
    cmap='coolwarm',
    square=True
    )
plt.suptitle("Correlation Matrix", y=1.02)
plt.tight_layout()
plt.show()
# Correlation matrix confirms positive strong correlations between SalePrice
# and Gr Liv Area (0.63), Total Bsmt SF (0.71) and Overall Qual (0.80).

# Analytical solution of linear regression
# Beta = (X^T X)^−1 X^Ty

# y = β0 + β1x1 + β2x2 + β3x3 + β4x4 + β5x5 + noise
feature_names = [
    'Gr Liv Area', 
    'Total Bsmt SF', 
    'Overall Qual', 
    'Central Air', 
    'Overall Cond'
    ]
X = df[feature_names].values
y = df['SalePrice'].values

# The column of ones is added for the constant intercept β0
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
# Solve the least squares problem (avoids matrix inversion)
# B, residuals, rank, s = np.linalg.lstsq(Xb, y, rcond=None)
# print("OLS Coefficients:", np.round(B, 3))
# uses matrix inversion:
XtX_inv = np.linalg.inv(Xb.T @ Xb)
B = XtX_inv @ Xb.T @ y
# Print intercept and coefficients
print(f'Intercept Manual OLS: {B[0]:.3f}')
print('Coefficients Manual OLS:')
for name, coef in zip(feature_names, B[1:]):
    print(f'{name:15}: {coef:.3f}')

# let's get predictions with the Hat matrix
H = Xb @ XtX_inv @ Xb.T
# in-sample predictions
y_pred_hat = H @ y 
# residuals
resid_hat = (np.eye(len(y)) - H) @ y

# OLS using statsmodels
ols = sm.OLS(y,Xb)
results = ols.fit()
# in-sample predictions
y_pred_sm = results.fittedvalues
# residuals
resid_sm = results.resid

# Crossplot: Predictions from Hat Matrix vs statsmodels
# a correct derivation shows points falling along a 45-degree line.
plt.figure(figsize=(7, 6))
plt.scatter(y_pred_hat, results.fittedvalues, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='y = x')
plt.xlabel('Predictions from Hat Matrix')
plt.ylabel('Predictions from statsmodels')
plt.title('Prediction Comparison')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Crossplot: Residuals from Hat Matrix vs statsmodels
plt.figure(figsize=(7, 6))
plt.scatter(resid_hat, results.resid, alpha=0.7, edgecolors='k')
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='y = x')
plt.xlabel('Residuals from Hat Matrix')
plt.ylabel('Residuals from statsmodels')
plt.title('Residual Comparison')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

print(results.summary())
# the 5th variable 'Overall Cond' is not statistically significant. P>|t| = 0.092 
# remove and re-define Xb
X = df[['Gr Liv Area', 
        'Total Bsmt SF', 
        'Overall Qual', 
        'Central Air'
        ]].values
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
ols = sm.OLS(y,Xb)
results = ols.fit()
results.summary()

# estimating OLS via scikit-learn
# no need to add column of ones by hand
feature_names = [
    'Gr Liv Area', 
    'Total Bsmt SF', 
    'Overall Qual', 
    'Central Air'
    ]
X = df[feature_names].values
y = df['SalePrice'].values

ols = LinearRegression()
ols.fit(X, y)

print('Intercept scikit-learn OLS:', ols.intercept_)
for name, coef in zip(feature_names, ols.coef_):
    print(f'{name:15}: {coef:.3f}')


# model predictions on fitted in-sample data
y_pred = ols.predict(X)
# Plot predicted y_pred vs actual SalePrice
plt.figure(figsize=(12, 10))
plt.scatter(y, y_pred, alpha=0.6, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # 45-degree line
plt.xlabel("Actual SalePrice (y)")
plt.ylabel("Predicted SalePrice (y_pred)")
plt.title("Predicted vs Actual SalePrice")
plt.grid(True)
plt.show()

# standardized residuals
# divide by standard deviation of y - y_pred
std_residuals = (y - y_pred) / np.std(y - y_pred)  

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, std_residuals, alpha=0.6, edgecolors='k')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.axhline(2, color='gray', linestyle='--')
plt.axhline(-2, color='gray', linestyle='--')
plt.xlabel("Predicted SalePrice (y_pred)")
plt.ylabel("Std Residuals")
plt.title("In-sample Std Residuals vs Predicted")
plt.grid(True)
plt.show()

# is the empirical risk (EE) a reliable estimate of Mean Integrated Square Error (MISE)?
# the empirical risk is the expectation of the error made by a linear model trained
# on data DN to predict the value of the output for the same dataset DN. Think of EE
# as training mean squared error or in-sample loss
# sigma^2 estimate
p = len(ols.coef_) + (1 if ols.fit_intercept else 0)
sg2_w = 1.0/(y.size - p) * np.sum((y - y_pred)**2)
print("sigma2_error estimate: ", sg2_w)
actual_EE = np.mean((y - y_pred)**2)
print("Actual Empirical Error: ", actual_EE)
expected_EE = (1.0 - (p/y.size))* sg2_w 
print("Expected Empirical Error: ", expected_EE)

# MISE estimate is derived from the sum of squared errors of a linear model trained
# on data DN and used to predict for the same training inputs X a set of testing outputs
# distributed according to a linear model. Training and testing set are independent.
# The estimate of the expected prediction error on unseen data is a good proxy for generalization error
MISE = (1.0 + (p/y.size)) * sg2_w
print("MISE estimate: ", MISE)

# ideally we would like standardized residuals to be distributed normally
# with mean 0 and standard deviation 1, contained within the interval [-2, 2]
# the U curve indicates that there is a systematic error in the predictions

# evaluating the performance of OLS on "unseen" out-of-sample data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
    )
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# plot a residual plot where we simply subtract the true target
# variables from our predicted responses:

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

std_residuals = (y_test - y_test_pred) / np.std(y_test - y_test_pred)

ax1.scatter(
    y_test_pred, std_residuals, 
    c='limegreen', marker='s',
    edgecolor='white', label='Test data'
    )

std_residuals = (y_train - y_train_pred) / np.std(y_train - y_train_pred)

ax2.scatter(
    y_train_pred, std_residuals,
    c='steelblue', marker='o', 
    edgecolor='white', label='Training data'
    )
ax1.set_ylabel('Residuals')

for ax in (ax1, ax2):
    ax.set_xlabel('Predicted values')
    ax.legend(loc='upper left')
    ax.hlines(0, xmin=x_min-100, xmax=x_max+100, color='red', linestyle='--', lw=2)
    ax.hlines(2, xmin=x_min-100, xmax=x_max+100, color='gray', linestyle='--')
    ax.hlines(-2, xmin=x_min-100, xmax=x_max+100, color='gray', linestyle='--')

    
plt.tight_layout()
plt.show()

"""
We should see residual plots for the test and training datasets with a line
passing through the x axis origin, the horizontal 0 value

In case of perfect predictions, the residuals would be exactly zero, 
which we will probably never encounter in realistic and practical applications. 
However, for a good regression model, we would expect the errors to be randomly 
distributed and the residuals to be randomly scattered around the centerline. 

If we see patterns in a residual plot, such as the U shape, it means that our model 
is unable to capture some explanatory information, like non-linear effects?
"""



