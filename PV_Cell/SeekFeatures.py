# import descriptor calculator
from rdkit import Chem
from mordred import Calculator, descriptors, error
from multiprocessing import freeze_support
# other modules used in this function
import pandas as pd
import numpy as np


def ChemFeatures(data):
    """
        This function is used to extract chemical information from
molecular structures in the form of Smiles_str.

    Parameters
    ----------
        data: array or array like of SMILES_str. 
              Input data to be analysed

    Returns
    ----------
        features_df: pd.DataFrame
                     a dataframe of features extracted from input SMILES_Str.
    """
    mols = [Chem.MolFromSmiles(mol) for mol in data]
    freeze_support()
    calc = Calculator(descriptors)
    print(list(calc.map(mols)))
    raw_data = calc.pandas(mols)
    features_df = pd.DataFrame(raw_data)
    return features_df


def Missingvaluelist(features_df):
    """
        This function loops through the dataframe and returns a list
of missing values with location

    Parameters
    ----------
        features_df: pd.DataFrame
                     a dataframe of features extracted from ChemFeatures.

    Returns
    ----------
        missing: an array or array like.
                 a dataframe of features extracted from input SMILES_Str.
    """
# find the non-value entries.
    missing =[]
    for i in range(features_df.shape[1]):
        if type(features_df.loc[1][i]) == error.Missing:
            missing.append(features_df.loc[1][i])

# show examples of the non-value entries.
    return missing


def ReplaceMissing(features_df):
    """
        This function automatically replaces all non-value entries(marked as
specific mordred.error.Missing message in the dataframe with a value 0.

    Parameters
    ----------
        features_df: pd.DataFrame
                     a dataframe of features extracted from ChemFeatures.

    Returns
    ----------
        features_df_n: pd.DataFrame
                       a dataframe of features with all missing values replaced
                       by 0.
    """
# first locate the columns that contain missing values
    features_df_n = features_df
    type_series = features_df_n.dtypes
    wrong_column = []
    for col in range(len(type_series)):
        if type_series[col] != np.float64 and type_series[col] != np.int64:
            wrong_column.append(col)

# use for loop within the columns found above and
# replace the missing values to 0
    for column in wrong_column:
        i = 0
        for item in features_df_n.iloc[:,column]:
            if type(item) != np.float64 and type(item) != np.int64:
                features_df_n.iloc[i,column] = 0
            i += 1
    return features_df_n


def seek_feature_with_replacement(data):
    """
    This function wrapps the three functions above, allowing users to extract features
    and replace invalid features in one step.
    
    Parameters
    ----------
        data: array or array like of SMILES_str. 
              Input data to be analysed

    Returns
    ----------
        features_df_n: pd.DataFrame
                       a dataframe of features with all missing values replaced
                       by 0.

    """
    features_df = ChemFeatures(data)
    features_df_n = ReplaceMissing(features_df)
    return features_df_n
