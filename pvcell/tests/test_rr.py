import pandas as pd
import numpy as np
import sys
sys.path.append("..")
import rrmodel
import processing
import seekfeatures as sf


def test_rr_regress():
    str = [{'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':1},
           {'SMILES_str':'[SiH2]1C=CC2=C1C=C([SiH2]2)C1=Cc2[se]ccc2[SiH2]1', 'pce':2},
           {'SMILES_str':'C1C=c2c3ccsc3c3[se]c4cc(oc4c3c2=C1)C1=CC=CC1', 'pce':3},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':4},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':5},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':6},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':7},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':8},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':9},
           {'SMILES_str':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1', 'pce':6}]
    data = pd.DataFrame(str)
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
