import pandas as pd
import numpy as np
import sys
sys.path.append("..")


def test_lasso_regress():
    data = pd.read_csv('../../Database/HCEPD_100K.csv')
    data = data.head(10)
    import seekfeatures as sf
    features_df = sf.seek_feature_with_replacement(data['SMILES_str'])
    import processing
    X, y = processing.scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = processing.kfold_sampling(X, y, n=5)
    import lassomodel
    coefs_LASSO, lambdas_LASSO, error1_LASSO, error2_LASSO, modelLASSO = \
        lassomodel.lasso_regress(X_train, y_train, a=X_test, b=y_test)
    assert len(lambdas_LASSO) == len(error1_LASSO)
    assert len(coefs_LASSO) == len(error2_LASSO)
    assert isinstance(coefs_LASSO, list)
    assert isinstance(error1_LASSO, list)
    assert isinstance(error2_LASSO, list)
    return 0
