import numpy as np

# regression models
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# plotting package
import matplotlib.pyplot as plt


def RRregress(X, y, a = None, b = None):
    """ 
    
        This function returns regression of input data by Ridge regression
        method.
        
    Parameters
    ----------
        X: an array or array-like predictors. 
           It should be scaled by StandardScaler.
        y: an array or array-like target. 
           It should has compatible dimension with input X.
        a, b: an array or array-like, optional.
           another set of data, such as a = X_test, b = y_test.

    Returns
    -------
        coefs_RR: list. 
                     a list of coefficients from RR with different lambdas
        lambdas_RR: list.
                       a list of lambdas used in this RR
        error1_RR: list.
                      a list of MSE of prediction from first input set (X, y)
        error2_RR: list.
                      a list of MSE of prediction from second input set (a, b). Return as None
                      if a and b are not defined.
        modelRR: modelRR = Ridge(), the LASSO model command
    """

    # RR vs lambda 
    coefs_RR = []
    error1_RR = []
    error2_RR = []
    # Tunning parameter(lambda)
    lambdas_RR = np.logspace(-4,8,200)
    modelRR = Ridge()
        
    # loop over lambda values to determine the best by mse
    for l in lambdas_RR:
        modelRR.set_params(alpha = l)
        modelRR.fit(X, y)
        coefs_RR.append(modelRR.coef_)
        error1_RR.append(mean_squared_error(y, modelRR.predict(X)))
        if a.any() and b.any() is not None:
            error2_RR.append(mean_squared_error(b, modelRR.predict(a)))
        else:
            error2_RR = None
    return coefs_RR, lambdas_RR, error1_RR, error2_RR, modelRR

def RR_plot(X, y, a = None, b = None):
    """ 
    
        This function returns regression of input data by RR regression 
        method.
    Parameters
    ----------
        X: an array or array-like predictors. 
           It should be scaled by StandardScaler.
        y: an array or array-like target. 
           It should has compatible dimension with input X.
        a, b: an array or array-like, optional.
           another set of data, such as a = X_test, b = y_test.

    Returns:
        two matplotlib plots.
        Left panel =  RRcoefs vs lambda, R panel = MSE vs lambda
    """

    # call for first return
    coefs_RR = RRregress(X, y, a, b)[0]
    lambdas_RR = RRregress(X, y, a, b)[1]
    error1_RR = RRregress(X, y, a, b)[2]
    error2_RR = RRregress(X, y, a, b)[3]
    fig = plt.figure(figsize=(12, 4.8))
    

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(lambdas_RR, coefs_RR)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('coefficients')
    plt.title('RR coefs vs $\lambda$')
    plt.xlim(1e-4, 1e8)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(lambdas_RR, error1_RR, label = 'train error')
    if error2_RR is not None:
        plt.plot(lambdas_RR, error2_RR, label = 'test error')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.xlim(1e-4, 1e8)
    plt.ylim(0, 6.5)
    plt.legend(loc = 'upper left')
    plt.title('MSE vs $\lambda$')