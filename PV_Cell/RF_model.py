from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
# plotting package
import matplotlib.pyplot as plt


def RFregress(X, y):
    """ 
    
        This function returns regression of input data by Random Forest 
        method. Need to call
        
    Parameters
    -----------
        X: array.
           a list of predictors processed by standard scaler.
        y: array.
           a list of target values(PCE from data)

    Returns
    -------
        modelRF: sklearn.ensemble.forest.RandomForestRegressor
                 the RF regression module.
        RF_fit: fitting result.
    """
    modelRF = RandomForestRegressor()
    RF_fit = modelRF.fit(X, y)
    return modelRF, RF_fit 


def RF_plot(X, y, a = None, b = None, name1 = 'data', name2 = None):
    """ 
    
        This function returns parity plot of regression result by Random
        Forest.
        
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
        matplotlib scatter plot.
        A parity plot that shows relationship between predicted values and actual values
        from train-set and test-set.
    """
    modelRF, RF_fit = RFregress(X, y)
    print("RF error for train set",mean_squared_error(y, modelRF.predict(X)))
    if a.any() and b.any() is not None:
        modelRF_test, RF_fit_test = RFregress(a, b)
        print("RF error for test set",mean_squared_error(b, modelRF.predict(a)))
    plt.figure(figsize=(8, 8)) 
    plt.scatter(y, RF_fit.predict(X), label = name1)
    if a.any() and b.any() is not None:
        plt.scatter(b, RF_fit_test.predict(a), label = name2)
    plt.plot([0,10], [0,10], lw = 4,color = 'black')
    plt.legend()
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    return
