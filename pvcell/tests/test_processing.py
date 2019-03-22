import pandas as pd
import numpy as np
import sys
sys.path.append("..")


def test_scale_sample():
    data = pd.read_csv('HCEPD_100K.csv')
    data = data.head(5)
    import seekfeatures as sf
    features_df = sf.chem_features(data['SMILES_str'])
    import processing
    X, y = processing.scale_sample(data, features_df)
    assert len(X) == len(y)
    assert len(X) == len(data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    return 0


def test_kfold_sampling():
    data = pd.read_csv('HCEPD_100K.csv')
    data = data.head(10)
    import seekfeatures as sf
    features_df = sf.chem_features(data['SMILES_str'])
    import processing
    X, y = processing.scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = processing.kfold_sampling(X, y, n=5)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    return 0
