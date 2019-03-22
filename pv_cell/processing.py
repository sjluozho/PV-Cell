"""
Pre-processing module.
"""
# Use StandardScaler to scale the features data
from sklearn.preprocessing import StandardScaler
# Use KFold as validation approach
from sklearn.model_selection import KFold


def scale_sample(data, features_df):
    """
        This function is used to scale the features to zero mean
        and 1 standard deviation.

    Parameters
    -----------
        data: pd.DataFrame.
            original data to be processed.
        features_df: pd.DataFrame.
                    a dataframe of features from seekfeatures.py
    Returns
    -------
        X: array.
           a list of predictors processed by standard scaler.
        y: array.
           a list of target values(PCE from data)
    """
    features = [feature for feature in features_df.columns]
    # Separating out the features
    x = features_df.loc[:, features].values
    # Our target is pce
    y = data['pce'].values
    sc = StandardScaler()
    X = sc.fit_transform(x)
    return X, y


def kfold_sampling(X, y, n=10):
    """
        This function applies k-fold cross-validation to
        the process of sampling.

    Parameters
    -----------
        X: an array or array-like.
           a list of predictors processed by standard scaler.
        y: an array or array-like.
           a list of target values(PCE from data)
    Returns
    -------
        X_train, X_test: array.
               a list of predictor train-set and a list of test set.
        y_train, y_test: array.
               a list of predictor train-set and a list of test set.
    """
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def sample_processor(data, features_df, n=10):
    """
        This function returns a train_set and test_set of data after
        k-fold cross-validation the process of sampling. The data are
        scaled by StandardScaler.

    Parameters
    -----------
        data: pd.DataFrame.
            original data to be processed.
        features_df: pd.DataFrame.
                    a dataframe of features from seekfeatures.py
        n : int.
            n_splits of KFold method.
    Returns
    -------
        X_train, X_test: array.
               a list of predictor train-set and a list of test set.
        y_train, y_test: array.
               a list of predictor train-set and a list of test set.

    """
    X, y = scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = kfold_sampling(X, y, n)
    return X_train, X_test, y_train, y_test
