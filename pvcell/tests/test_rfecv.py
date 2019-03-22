import pandas as pd
import numpy as np
import sklearn
import sys
sys.path.append("../")


def test_rfecv_regress():
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
    import seekfeatures as sf
    features_df = sf.seek_feature_with_replacement(data['SMILES_str'])
    import processing
    X, y = processing.scale_sample(data, features_df)
    import rfecvmodel
    modelRFECV = rfecvmodel.rfecv_regress(X, y, step=5,
                                          estimator='linear',
                                          scoring_method='explained_variance')
    coef_list = [x for x in modelRFECV.estimator_.coef_[0]]
    assert isinstance(coef_list, list)
    assert len(coef_list) < 1825
    assert isinstance(modelRFECV, sklearn.feature_selection.rfe.RFECV)
    return 0
