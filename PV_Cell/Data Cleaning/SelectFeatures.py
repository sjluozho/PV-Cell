# Module for extracting ChemInfo
from SeekFeatures import *

# other modules required in this function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Use StandardScaler to scale the features data
from sklearn.preprocessing import StandardScaler
features =[feature for feature in features_df.columns]

# Use KFold as validation approach
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# regression models
from sklearn.linear_model import Ridge

def Scale_sample(data, features_df):
    """
        This function is used to scale the features to zero mean 
        and 1 standard deviation..

    Attributes:
        data: original data. a pd.DataFrame.
        features_df: a pd.DataFrame of features from SeekFeatures

    """
    features =[feature for feature in features_df.columns]
    # Separating out the features
    x = features_df.loc[:, features].values
    # Our target is pce
    y = data['pce'].values
    sc = StandardScaler()
    X = sc.fit_transform(x)
    return X, y


def KFold_sampling(X, y)
    """
        This function applies k-fold cross-validation to 
        the process of sampling.

    Attributes:
        X: an array or array-like of features
        y: an array or array-like of pce data

    """
    kf = KFold(n_splits = 10, shuffle = True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


