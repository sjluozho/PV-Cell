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
    Attribute:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.
        **kwarg: input a different set of data. *Format* a = X_test, b = y_test.

    Returns:
        coefs_RR: a 2D list of coefficients from RR with different lambdas
        lambdas_RR: a list of lambdas used in this RR
        error1_RR: MSE of prediction from first input set (X, y)
        error2_RR: MSE of prediction from second input set (a, b). Return as None
                   if a and b are not defined.
        modelRR: modelRR = Ridge(), the Ridge model command
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
        if a and b is not None:
            error2_RR.append(mean_squared_error(b, modelRR.predict(a)))
        else:
            error2_RR = None
    return coefs_RR, lambdas_RR, error1_RR, error2_RR, modelRR

def RR_plot(X, y, a = None, b = None):
    """ 
    
        This function returns regression of input data by Ridge regression 
        method.
    Attribute:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.
        **kwarg: input a different set of data. *Format* a = X_test, b = y_test.
                Pass if a and b are not defined.
    Returns:
        two plots: Left panel =  RR coefs vs lambda, R panel = MSE vs lambda
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