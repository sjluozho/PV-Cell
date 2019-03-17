import numpy as np
# regression models
from sklearn.linear_model import Ridge


def RRregress(X_train, X_test, y_train, y_test):
    
    # RR vs lambda 
    coefs_RR = []
    train_error_RR = []
    test_error_RR = []
    # Tunning parameter(lambda)
    lambdas_RR = np.logspace(-4,8,200)
    modelRR = Ridge()

    # loop over lambda values to determine the best by mse
    for l in lambdas_RR:
        modelRR.set_params(alpha = l)
        modelRR.fit(X_train, y_train)
        coefs_RR.append(modelRR.coef_)
        train_error_RR.append(mean_squared_error(y_train, modelRR.predict(X_train)))
        test_error_RR.append(mean_squared_error(y_test, modelRR.predict(X_test)))
    return coefs_RR, lambdas_RR, test_error_RR, modelRR

def RRplot(X_train, X_test, y_train, y_test):
    coefs_RR, lambdas_RR, test_error_RR = RR_regress(X_train, X_test, y_train, y_test)
    
    fig = plt.figure(figsize=(12, 4.8))

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(lambdas_RR, coefs_RR)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('coefficients')
    plt.title('RR coefs vs $\lambda$')
    plt.xlim(1e-4, 1e8)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(lambdas_RR, train_error_RR, label = 'train error')
    plt.plot(lambdas_RR, test_error_RR, label = 'test error')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.xlim(1e-4, 1e8)
    plt.ylim(0, 6.5)
    plt.legend(loc = 'upper left')
    plt.title('MSE vs $\lambda$')