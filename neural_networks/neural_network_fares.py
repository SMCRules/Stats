import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def data_load(file_path):
    # Identify the dataset based on the file name
    file_name = file_path.split('/')[-1]
    df = pd.read_csv(file_path)
    
    if 'breast-cancer' in file_name:        
        # Drop the redundant 'id' column
        df.drop('id', axis=1, inplace=True)
        # Encode the label into binary (0/1)
        df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
        # Compute correlations to remove weakly correlated features
        corr = df.corr()
        cor_target = abs(corr["diagnosis"])
        # Select highly correlated features (threshold = 0.25)
        relevant_features = cor_target[cor_target > 0.25].index
        df = df[relevant_features]
        X_cols = df.drop('diagnosis', axis=1).columns
        X = df.drop('diagnosis', axis=1).values
        y = df['diagnosis'].values.reshape(-1,1)
        
    elif 'diabetes' in file_name:
        X_cols = df.drop('Outcome', axis=1).columns       
        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values.reshape(-1,1)

    elif 'Iris' in file_name:        
        # Drop the 'Id' column
        df.drop('Id', axis=1, inplace=True)
        # Encode the target variable with LabelEncoder
        le = LabelEncoder()
        df['Species'] = le.fit_transform(df['Species'])
        X_cols = df.drop('Species', axis=1).columns
        X = df.drop('Species', axis=1).values
        y = df['Species'].values.reshape(-1,1)

    else:
        raise ValueError("Unsupported dataset. Please provide a valid file path.")
    
    return X, y, X_cols

def scale(X):
    """
    Standardizes the data in the array X.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized features array.
    """
    # Calculate the mean and standard deviation of each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Standardize the data
    X = (X - mean) / std

    return X

def train_test_split(X, y, random_state=41, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).
        y (numpy.ndarray): Target array of shape (n_samples,).
        random_state (int): Seed for the random number generator. Default is 42.
        test_size (float): Proportion of samples to include in the test set. Default is 0.2.

    Returns:
        Tuple[numpy.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
    """
    # Get number of samples
    n_samples = X.shape[0]

    # Set the seed for the random number generator
    np.random.seed(random_state)

    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))

    # Determine the size of the test set
    test_size = int(n_samples * test_size)

    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    # Split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test   



X, y, X_cols = data_load('/home/miguel/Python_Projects/datasets/breast-cancer.xls')
print(X.shape, y.shape)

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

