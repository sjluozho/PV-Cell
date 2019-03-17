import pandas as pd
# import descriptor calculator
from rdkit import Chem
from mordred import Calculator, descriptors
from multiprocessing import freeze_support


def ChemFeatures(data):
    """
        This function is used to extract chemical information from
molecular structures in the form of Smiles_str.

    Attributes:
        data: array or array like of SMILES_str. Input data to be
analysed

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

   Attributes:
	features_df: a pd.DataFrame from the function ChemFeatures"""
# find the non-value entries.
missing =[]
for i in range(f.shape[1]):
    if type(features_df.loc[1][i]) == mordred.error.Missing:
        missing.append(features_df.loc[1][i])

# show examples of the non-value entries.
	return missing


def ReplaceMissing(features_df):
    """
	This function automatically replaces all non-value entries(marked as
	specific mordred.error.Missing message in the dataframe with a value 0.

    Attributes:
	features_df: a pd.DataFrame containing features info of all molecules"""

	# first locate the columns that contain missing values
	feature_df_n = features_df
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
