import pandas as pd
import numpy as np
import sklearn
import sys
sys.path.append("..")
import seekfeatures as sf
import processing
import rfmodel


def test_lasso_regress():
    data = pd.read_csv('../../Database/HCEPD_100K.csv')
    data = data.head(10)
    features_df = sf.seek_feature_with_replacement(data['SMILES_str'])
    X, y = processing.scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = processing.kfold_sampling(X, y, n=5)
    modelRF, RF_fit =  rfmodel.rf_regress(X, y)
    assert len(modelRF) == 10
    assert len(RF_fit) == 10
    assert isinstance(modelRF, sklearn.ensemble.forest.RandomForestRegressor)
    assert isinstance(RF_fit, sklearn.ensemble.forest.RandomForestRegressor)
    return 0
