from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

import matplotlib.pyplot as plt


# It seems only to be compatible with linear model.
def rfecv_regress(X, y, step=5, estimator='linear',
                 scoring_method='explained_variance'):
    """

        This function returns regression of input data by RFECV(recursive
        feature elimination by cross-validation).

    Parameters
    ----------
        X: an array or array-like predictors.
           It should be scaled by StandardScaler.
        y: an array or array-like target.
           It should has compatible dimension with input X.
        a, b: an array or array-like, optional.
           another set of data, such as a = X_test, b = y_test.
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
        A plot of cross_val score vs number of features selected. And optimal
        number of features given by this method.
    """

    estimatorRFECV = SVR(estimator)
    modelRFECV = RFECV(estimatorRFECV, step=5, scoring=scoring_method)
    modelRFECV.fit(X, y)

    print("Optimal number of features : %d" % modelRFECV.n_features_)
    print(modelRFECV.support_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(modelRFECV.grid_scores_) + 1),
             modelRFECV.grid_scores_)
    plt.show()
    return modelRFECV
