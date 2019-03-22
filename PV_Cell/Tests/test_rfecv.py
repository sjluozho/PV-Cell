import pandas as pd
import numpy as np
import sklearn
import sys
sys.path.append("..")
import seekfeatures as sf
import processing
import rfecvmodel


def test_lasso_regress():
    data = pd.read_csv('../../Database/HCEPD_100K.csv')
    data = data.head(10)
    features_df = sf.seek_feature_with_replacement(data['SMILES_str'])
    X, y = processing.scale_sample(data, features_df)
    modelRFECV = rfecvmodel.rfecv_regress(X, y, step = 5,
                                          estimator = 'linear',
                                          scoring_method='explained_variance' )
    coef_list = [x for x in modelRFECV.estimator_.coef_[0]]
    assert isinstance(coef_list, list)
    assert len(coef_list) < 1825
    assert isinstance(modelRFECV, sklearn.feature_selection.rfe.RFECV)
    return 0
