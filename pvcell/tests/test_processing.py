import pandas as pd
import numpy as np
import sys
sys.path.append("../")


def test_scale_sample():
    str = [{'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1}]
    data = pd.DataFrame(str)
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
    str = [{'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1},
           {'SMILES_str': 'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1',
            'pce': 1}]
    data = pd.DataFrame(str)
    import seekfeatures as sf
    features_df = sf.chem_features(data['SMILES_str'])
    import processing
    X, y = processing.scale_sample(data, features_df)
    X_train, X_test, y_train, y_test = processing.kfold_sampling(X, y, n=5)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    return 0
