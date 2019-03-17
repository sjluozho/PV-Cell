# Module for extracting ChemInfo
from SeekFeatures import *
from Processing import *

# import Modules of all four regression models
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from RF_model import *
from RFECV_model import *

# other modules required in this function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def Selector(data, features_df, lam = [1e2, 1e2], n = 10, estimator = 'rbf', scoring_method='explained_variance'):
    # split data into a train_set and test_Set
    X_train, X_test, y_train, y_test = SampleProcessor(data, features_df, n = 10)
    # regress by Ridge
    modelRR = Ridge()
    modelRR_lam = modelRR.set_params(alpha = lam[0])
    # regress by LASSO
    modelLASSO = Lasso()
    modelLASSO_lam = modelLasso.set_params(alpha = lam[1])
    # regress by Ramdon Forest
    RF_train, RF_train_fit = RFregress(X_train, y_train)
    RF_test, RF_test_fit = RFregress(X_test, y_test)
    # regress by RFECV
    modelRFECV = RFECVregress(X, y, estimator = 'rbf', scoring_method='explained_variance')
    
    # make parity plot
    fig = plt.figure(figsize=(12, 12)) 

    ax1 = plt.subplot(2, 2, 1)
    #plt.xlim([0,50]);
    #plt.ylim([0,50]);
    plt.scatter(y_train, modelRR_lam.predict(X_train), label = 'Training')
    plt.scatter(y_test, modelRR_lam.predict(X_test),color = 'r', label = 'Test')
    plt.plot([0,10], [0,10], lw = 4,color = 'black')
    plt.legend()
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.title('RR Parity Plot')

    ax2 = plt.subplot(2, 2, 2)
    #plt.xlim([0,50]);
    #plt.ylim([0,50]);
    plt.scatter(y_train, modelLASSO_lam.predict(X_train), label = 'Training')
    plt.scatter(y_test, modelLASSO_lam.predict(X_test), color = 'r', label = 'Test')
    plt.plot([0,10], [0,10], lw = 4,color = 'black')
    plt.legend()
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.title('RR Parity Plot')

    ax3= plt.subplot(2, 2, 3)
    #plt.xlim([0,50]);
    #plt.ylim([0,50]);
    plt.scatter(y_train, RF_train_fit.predict(X_train), label = 'Training')
    plt.scatter(y_test, RF_train_fit.predict(X_test), color = 'r', label = 'Test')
    plt.plot([0,10], [0,10], lw = 4,color = 'black')
    plt.legend()
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.title('RF Parity Plot')

    ax4 = plt.subplot(2, 2, 4)
    #plt.xlim([0,50]);
    #plt.ylim([0,50]);
    plt.scatter(y_train, modelRFECV.predict(X_train), label = 'Training')
    plt.scatter(y_test, modelRFECV.predict(X_test), color = 'r', label = 'Test')
    plt.plot([0,10], [0,10], lw = 4, color = 'black')
    plt.legend()
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.title('RFECV Parity Plot')
