from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
# plotting package
import matplotlib.pyplot as plt


def RFregress(X, y):
    """ 
    
        This function returns regression of input data by Random Forest 
        method. Need to call
        
    Attributes:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.

    """
    modelRF = RandomForestRegressor()
    RF_fit = modelRF.fit(X, y)
    return modelRF, RF_fit 


def RF_plot(X, y, a = None, b = None, name1 = 'data', name2 = None):
    """ 
    
        This function returns parity plot of regression result by Random
        Forest. 
    Attribute:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.
        name: str or str-like. It indicates the categoric of input data(train 
              _set or test_set).
        a, b : different set of data. None as default.
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