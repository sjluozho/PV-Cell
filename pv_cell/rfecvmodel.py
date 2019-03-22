from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns


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


def snsplot(X, y, features_df):
    """

        This function returns feature coef plot by seanborn.

    Parameters
    ----------
        X: an array or array-like predictors.
           It should be scaled by StandardScaler.
        y: an array or array-like target.
           It should has compatible dimension with input X.
        features_df: pd.DataFrame.
                     feature dataframe.

    Returns:
        a seaborn plot.
        A plot that shows features and their corresponding coef.
    """

    modelRFECV = rfecv_regress(X, y)
    RFECVrank = modelRFECV.ranking_
    features60 = pd.DataFrame({'Feature': eatures_df.columns.values,
                               'Rank': RFECVrank}).sort_values('Rank')
    features_title = features60.Feature[0:60].values

    # Compile into a dataframe
    RFECV_df = pd.DataFrame()
    RFECV_df['coef'] = coef_list
    RFECV_df['Feature'] = [x for x in features_title]

    # Put the mean scores into a Pandas dataframe
    meanplot = pd.DataFrame({'Feature': RFECV_df.Feature,
                             'Coef': RFECV_df.coef})

    # Sort the dataframe
    meanplot = meanplot.sort_values('Coef', ascending=False)

    # Plot by seaborn
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    coef_plot = sns.factorplot(y='Coef', x='Feature',
                               data=meanplot, kind="bar",
                               size=14, aspect=1.9, orient="v",
                               palette='coolwarm')
    coef_plot.set_xticklabels(rotation=90, size=20)
    coef_plot.set_yticklabels(size=20)
