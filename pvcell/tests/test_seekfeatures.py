import pandas as pd
import sys
sys.path.append("..")


def test_chemfeatures():
    str = [{'a':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1'},
           {'a':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1'}]
    data = pd.DataFrame(str)
    import seekfeatures as sf
    generated_features = sf.chem_features(data['a'])
    assert type(generated_features) == pd.core.frame.DataFrame
    assert len(generated_features) == len(data)
    return 0


def test_missingvaluelist():
    str = [{'a':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1'},
           {'a':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1'}]
    data = pd.DataFrame(str)
    import seekfeatures as sf
    generated_features = sf.chem_features(data['a'])
    generated_list = sf.missing_value_list(generated_features)
    assert isinstance(generated_list, list)
    return 0


def test_replacemissing():
    str = [{'a':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1'},
           {'a':'C1C=CC=C1c1cc2[se]c3c4occc4c4nsnc4c3c2cn1'}]
    data = pd.DataFrame(str)
    import seekfeatures as sf
    generated_features = sf.chem_features(data['a'])
    effective_feature = sf.replace_missing(generated_features)
    assert type(effective_feature) == pd.core.frame.DataFrame
    assert generated_features.shape == effective_feature.shape
    assert sf.missing_value_list(effective_feature) == []
    return 0
