import pandas as pd
import sys
sys.path.append("..")


def test_chemfeatures():
    data = pd.read_csv('../../Database/HCEPD_100K.csv')
    data = data.head(5)
    import seekfeatures as sf
    generated_features = sf.chem_features(data['SMILES_str'])
    assert type(generated_features) == pd.core.frame.DataFrame
    assert len(generated_features) == len(data)
    return 0


def test_missingvaluelist():
    data = pd.read_csv('../../Database/HCEPD_100K.csv')
    data = data.head(5)
    import seekfeatures as sf
    generated_features = sf.chem_features(data['SMILES_str'])
    generated_list = sf.missing_value_list(generated_features)
    assert isinstance(generated_list, list)
    assert 'mordred.error.Missing' in str(generated_list)
    return 0


def test_replacemissing():
    data = pd.read_csv('../../Database/HCEPD_100K.csv')
    data = data.head(5)
    import seekfeatures as sf
    generated_features = sf.chem_features(data['SMILES_str'])
    effective_feature = sf.replace_missing(generated_features)
    assert type(effective_feature) == pd.core.frame.DataFrame
    assert generated_features.shape == effective_feature.shape
    assert sf.missing_value_list(effective_feature) == []
    return 0
