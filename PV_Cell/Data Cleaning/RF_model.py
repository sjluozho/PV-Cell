from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
# plotting package
import matplotlib.pyplot as plt


def RFregress(X, y):
    """ 
    
        This function returns regression of input data by Random Forest 
        method. Need to call
    Attribute:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.

    """
    modelRF = RandomForestRegressor()
    RF_fit = modelRF.fit(X, y)
    return modelRF, RF_fit 


def RF_plot(X, y, name = 'data'):
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

    """
    modelRF, RF_fit = RFregress(X, y)
    print("RF error",mean_squared_error(y, modelRF.predict(X)))
    plt.figure(figsize=(8, 8)) 
    plt.scatter(y, RF_fit.predict(X), label = name)
    plt.plot([0,10], [0,10], lw = 4,color = 'black')
    plt.legend()
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    return