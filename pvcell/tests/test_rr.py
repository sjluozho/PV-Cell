import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import rrmodel
import processing
import seekfeatures as sf


def test_rr_regress():
    data = pd.read_csv('HCEPD_100K.csv')
    data = data.head(10)
    features_df = sf.seek_feature_with_replacement(data['SMILES_str'])
    X, y = processing.scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = processing.kfold_sampling(X, y, n=5)
    coefs_RR, lambdas_RR, error1_RR, error2_RR, modelRR = rrmodel.rr_regress(
        X_train, y_train, a=X_test, b=y_test)
    assert len(lambdas_RR) == len(error1_RR)
    assert len(coefs_RR) == len(error2_RR)
    assert isinstance(coefs_RR, list)
    assert isinstance(error1_RR, list)
    assert isinstance(error2_RR, list)
    return 0
