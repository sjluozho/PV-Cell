import numpy as np
# regression models
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# plotting package
import matplotlib.pyplot as plt


def LASSOregress(X_train, X_test, y_train, y_test):
    
    # RR vs lambda 
    coefs_LASSO = []
    train_error_LASSO = []
    test_error_LASSO = []
    # Tunning parameter(lambda)
    lambdas_LASSO = np.logspace(-4,8,200)
    modelLASSO = Lasso(max_iter = 1e4)

    # loop over lambda values to determine the best by mse
    for l in lambdas_LASSO:
        modelLASSO.set_params(alpha = l)
        modelLASSO.fit(X_train, y_train)
        coefs_LASSO.append(modelLASSO.coef_)
        train_error_LASSO.append(mean_squared_error(y_train, modelLASSO.predict(X_train)))
        test_error_LASSO.append(mean_squared_error(y_test, modelLASSO.predict(X_test)))
    return coefs_LASSO, lambdas_LASSO, train_error_LASSO, test_error_LASSO, modelLASSO

def LASSOplot(X_train, X_test, y_train, y_test):
    coefs_LASSO, lambdas_LASSO, train_error_LASSO, test_error_LASSO, modelLASSO = LASSOregress(X_train, X_test, y_train, y_test)
    
    fig = plt.figure(figsize=(12, 4.8))

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(lambdas_LASSO, coefs_LASSO)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('coefficients')
    plt.title('LASSO coefs vs $\lambda$')
    plt.xlim(1e-4, 1e8)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(lambdas_LASSO, train_error_LASSO, label = 'train error')
    plt.plot(lambdas_LASSO, test_error_LASSO, label = 'test error')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.xlim(1e-4, 1e8)
    plt.ylim(0, 6.5)
    plt.legend(loc = 'upper left')
    plt.title('MSE vs $\lambda$')