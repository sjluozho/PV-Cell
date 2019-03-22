import pandas as pd
import numpy as np
import sklearn
import sys
sys.path.append("..//")


def test_rf_regress():
    str = [{'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1}]
    data = pd.DataFrame(str)
    import seekfeatures as sf
    features_df = sf.seek_feature_with_replacement(data['SMILES_str'])
    import processing
    X, y = processing.scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = processing.kfold_sampling(X, y, n=5)
    import rfmodel
    modelRF, RF_fit = rfmodel.rf_regress(X, y)
    assert len(modelRF) == 10
    assert len(RF_fit) == 10
    assert isinstance(modelRF, sklearn.ensemble.forest.RandomForestRegressor)
    assert isinstance(RF_fit, sklearn.ensemble.forest.RandomForestRegressor)
    return 0
