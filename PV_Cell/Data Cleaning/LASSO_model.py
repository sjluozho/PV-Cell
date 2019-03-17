import numpy as np

# regression models
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# plotting package
import matplotlib.pyplot as plt


def LASSOregress(X, y, a = None, b = None):
    """ 
    
        This function returns regression of input data by LASSO regression
        method.
    Attributes:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.
        **kwarg: input a different set of data. *Format* a = X_test, b = y_test.

    Returns:
        coefs_LASSO: a 2D list of coefficients from LASSO with different lambdas
        lambdas_LASSO: a list of lambdas used in this LASSO
        error1_LASSO: MSE of prediction from first input set (X, y)
        error2_LASSO: MSE of prediction from second input set (a, b). Return as None
                   if a and b are not defined.
        modelLASSO: modelLASSO = Lasso(), the LASSO model command
    """

    # lASSO vs lambda 
    coefs_LASSO = []
    error1_LASSO = []
    error2_LASSO = []
    # Tunning parameter(lambda)
    lambdas_LASSO = np.logspace(-4,8,200)
    modelLASSO = Lasso()
        
    # loop over lambda values to determine the best by mse
    for l in lambdas_LASSO:
        modelLASSO.set_params(alpha = l)
        modelLASSO.fit(X, y)
        coefs_LASSO.append(modelRR.coef_)
        error1_LASSO.append(mean_squared_error(y, modelLASSO.predict(X)))
        if a and b is not None:
            error2_LASSO.append(mean_squared_error(b, modelLASSO.predict(a)))
    return coefs_LASSO, lambdas_RR, error1_RR, error2_RR, modelRR

def LASSO_plot(X, y, a = None, b = None):
    """ 
    
        This function returns regression of input data by LASSO regression 
        method.
    Attribute:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.
        **kwarg: input a different set of data. *Format* a = X_test, b = y_test.
                Pass if a and b are not defined.
    Returns:
        two plots: Left panel =  LASSO coefs vs lambda, R panel = MSE vs lambda
    """

    # call for first return
    coefs_LASSO = LASSOregress(X, y, a, b)[0]
    lambdas_LASSO = LASSOregress(X, y, a, b)[1]
    error1_LASSO = LASSOregress(X, y, a, b)[2]
    error2_LASSO = LASSOregress(X, y, a, b)[3]
    fig = plt.figure(figsize=(12, 4.8))
    

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(lambdas_LASSO, coefs_LASSO)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('coefficients')
    plt.title('LASSO coefs vs $\lambda$')
    plt.xlim(1e-4, 1e8)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(lambdas_RR, error1_LASSO, label = 'train error')
    if error2_RR is not None:
        plt.plot(lambdas_RR, error2_LASSO, label = 'test error')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.xlim(1e-4, 1e8)
    plt.ylim(0, 6.5)
    plt.legend(loc = 'upper left')
    plt.title('MSE vs $\lambda$')