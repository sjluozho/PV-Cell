# Module for extracting ChemInfo
from seekfeatures import *
from processing import *

# import Modules of all four regression models
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from rfmodel import *
from rfecvmodel import *

# other modules required in this function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


def selector(data, features_df, lam=[1e2, 1e2], n=10,
             estimator='linear', scoring_method='explained_variance'):
    """

        This function is a wrap-up function that help users determine which
        mthod to use for features reduction.

    Parameters
    ----------
        data: pd.DataFrame
              Original data frame to be processed.
        features_df: pd.DataFrame
                     features dataframe after seekfeature.
        lam: array(1X2).
             lam for Lasso and Ridge regression. Users need to define a good
             one from other module rrmodel or lassomodel.
        n: int.
           kfold splitnumber
        estimator: str.
                   Specifies the kernel type to be used in the algorithm.
                   'Radial basis function' as default.
        scoring_method: str.
                        scoring_method for cross-validation.
                        'explained_variance'
                        as default. For more availalbe methods, refer to :
            https://scikit-learn.org/stable/modules/model_evaluation.html

    Returns:
        a matplotlib plot.
        For each method, generate a parity plot.
    """

    # split data into a train_set and test_Set
    X_train, X_test, y_train, y_test = sample_processor(data, features_df,
                                                        n=10)
    # regress by Ridge
    modelRR = Ridge()
    modelRR_lam = modelRR.set_params(alpha=lam[0])
    modelRR_lam_train = modelRR_lam.fit(X_train, y_train)
    modelRR_lam_test = modelRR_lam.fit(X_test, y_test)
    # regress by LASSO
    modelLASSO = Lasso()
    modelLASSO_lam = modelLASSO.set_params(alpha=lam[1])
    modelLASSO_lam_train = modelLASSO_lam.fit(X_train, y_train)
    modelLASSO_lam_test = modelLASSO_lam.fit(X_test, y_test)
    # regress by Ramdon Forest
    RF_train, RF_train_fit = rf_regress(X_train, y_train)
    RF_test, RF_test_fit = rf_regress(X_test, y_test)
    # regress by RFECV
    modelRFECV = rfecv_regress(X, y, estimator='linear',
                               scoring_method='explained_variance')

    # make parity plot
    fig = plt.figure(figsize=(12, 12))
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    # set appropriate distances between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1 = plt.subplot(2, 2, 1)
    plt.scatter(y_train, modelRR_lam.predict(X_train),
                label='Training')
    plt.scatter(y_test, modelRR_lam.predict(X_test), color='r',
                label='Test')
    plt.plot([0, 10], [0, 10], lw=4, color='black')
    plt.legend(fontsize=10)
    plt.xlabel('Actual Output', fontsize=10)
    plt.ylabel('Predicted Output', fontsize=10)
    plt.title('RR Parity Plot', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax2 = plt.subplot(2, 2, 2)
    plt.scatter(y_train, modelLASSO_lam.predict(X_train),
                label='Training')
    plt.scatter(y_test, modelLASSO_lam.predict(X_test), color='r',
                label='Test')
    plt.plot([0, 10], [0, 10], lw=4, color='black')
    plt.legend(fontsize=10)
    plt.xlabel('Actual Output', fontsize=10)
    plt.ylabel('Predicted Output', fontsize=10)
    plt.title('RR Parity Plot', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax3 = plt.subplot(2, 2, 3)
    plt.scatter(y_train, RF_train_fit.predict(X_train),
                label='Training')
    plt.scatter(y_test, RF_train_fit.predict(X_test), color='r',
                label='Test')
    plt.plot([0, 10], [0, 10], lw=4, color='black')
    plt.legend(fontsize=10)
    plt.xlabel('Actual Output', fontsize=10)
    plt.ylabel('Predicted Output', fontsize=10)
    plt.title('RF Parity Plot', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax4 = plt.subplot(2, 2, 4)
    plt.scatter(y_train, modelRFECV.predict(X_train), label='Training')
    plt.scatter(y_test, modelRFECV.predict(X_test), color='r',
                label='Test')
    plt.plot([0, 10], [0, 10], lw=4, color='black')
    plt.legend(fontsize=10)
    plt.xlabel('Actual Output', fontsize=10)
    plt.ylabel('Predicted Output', fontsize=10)
    plt.title('RFECV Parity Plot', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    return
