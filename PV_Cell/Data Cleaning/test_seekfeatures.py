def test_ChemFeatures():
    """"""
    import SeekFeatures
    import pandas as pd
    data = pd.read_csv('../Database/HCEPD_100K.csv') 
    data = data.head(5)
    generated_features = SeekFeatures.ChemFeatures(data['SMILES_str'])
    assert type(generated_features) == pd.core.frame.DataFrame
    assert len(generated_features) == len(data)
    return 0


def test_Missingvaluelist():
    import SeekFeatures
    import pandas as pd
    data = pd.read_csv('../Database/HCEPD_100K.csv')
    data = data.head(5)
    feature_df = SeekFeatures.ChemFeatures(data['SMILES_str'])
    generated_list = SeekFeatures.Missingvaluelist(feature_df)
    assert isinstance(generated_list, list) 
    assert 'mordred.error.Missing' in str(generated_list)
    return 0


def test_ReplaceMissing():
    import pandas as pd
    import SeekFeatures
    data = pd.read_csv('../Database/HCEPD_100K.csv')
    data = data.head(5)
    feature_df = SeekFeatures.ChemFeatures(data['SMILES_str'])
    effective_feature = SeekFeatures.ReplaceMissing(feature_df)
    assert type(effective_feature) == pd.core.frame.DataFrame
    assert feature_df.shape == effective_feature.shape
    assert SeekFeatures.Missingvaluelist(effective_feature) == []
    return 0
