from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


def RFECVregress(X, y, estimator = 'rbf', scoring_method='explained_variance' ):
    """ 
    
        This function returns regression of input data by RFECV(recursive feature elimination
        by cross-validation).
        
    Attributes:
        X: an array or array-like predictors. It should be scaled by
           StandardScaler.
        y: an array or array-like target. It should has compatible dimension
           with input X.
        estimator: Specifies the kernel type to be used in the algorithm.
                  'Radial basis function' as default.
        scoring_method: scoring_method for cross-validation. 'explained_variance'
                        as default. For more availalbe methods, refer to :
            https://scikit-learn.org/stable/modules/model_evaluation.html

    Returns:
        A plot of cross_val score vs number of features selected. And optimal number
        of features given by this method.
    """
    
    estimatorREFCV = SVR(estimator)
    modelREFCV = RFECV(estimatorREFCV, step=5, scoring= scoring_method)
    modelREFCV.fit(X, y)

    print("Optimal number of features : %d" % modelREFCV.n_features_)
    print(modelREFCV.support_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(modelREFCV.grid_scores_) + 1), modelREFCV.grid_scores_)
    plt.show()
