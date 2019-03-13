def SeekFeatures(data):
    """
        This function is used to extract chemical information from
molecular structures in the form of Smiles_str.

    Attributes:
        data: array or array like. Input data to be analysed

    """
    freeze_support()
    calc = Calculator(descriptors)
    print(list(calc.map(mols)))
    raw_data = calc.pandas(mols)
    features_df = pd.DataFrame(raw_data)
    return features_df
