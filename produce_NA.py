# import packages
# todo!pip install wget
import copy

import wget
# todo!wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')
from utils import *
import torch
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import missingno as msgn
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import json
from scipy.stats import ks_2samp


# global variables
data_address = 'data/complete_dataset.csv'
p_miss = 0.4
p_obs = 0.5


class MissingData:
    """MissingData is a helper class that allows produce missing data from complete dataset, impute data by ML
    algorithms, and compares distributions.
        Attributes:
            None.
        Methods:
            data_generation : csv file address located at data to numpy datatype
            produce_NA: Generate missing values for specifics missing-data mechanism and proportion of missing values.
        """

    @staticmethod
    def data_generation(data_address):
        """complete data address located at data directory to numpy datatype"""
        data_pd = pd.read_csv(data_address)
        columns = list(data_pd.columns)
        data_np = data_pd.to_numpy()
        return data_pd, data_np, columns

    @staticmethod
    def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
        """
        Generate missing values for specifics missing-data mechanism and proportion of missing values.

        Parameters
        ----------
        X : torch.DoubleTensor or np.ndarray, shape (n, d)
            Data for which missing values will be simulated.
            If a numpy array is provided, it will be converted to a pytorch tensor.
        p_miss : float
            Proportion of missing values to generate for variables which will have missing values.
        mecha : str,
                Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
        opt: str,
             For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
        p_obs : float
                If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
        q : float
            If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

        Returns
        ----------
        A dictionnary containing:
        'X_init': the initial data matrix.
        'X_incomp': the data with the generated missing values.
        'mask': a matrix indexing the generated missing values.s
        """

        to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
        if not to_torch:
            X = X.astype(np.float32)
            X = torch.from_numpy(X)

        if mecha == "MAR":
            mask = MAR_mask(X, p_miss, p_obs).double()
        elif mecha == "MNAR" and opt == "logistic":
            mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
        elif mecha == "MNAR" and opt == "quantile":
            mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
        elif mecha == "MNAR" and opt == "selfmasked":
            mask = MNAR_self_mask_logistic(X, p_miss).double()
        else:
            mask = (torch.rand(X.shape) < p_miss).double()

        X_nas = X.clone()
        X_nas[mask.bool()] = np.nan

        return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}

    @staticmethod
    def get_three_missing_datasets(complete_data,columns):
        """get three types of missing datasets and store all in dictionary
        arguments:
            complete_data: numpy
            columns: list of column names that used for pandas dataframe
        return:
            Dictionary of pandas dataframe
        """
        missing_mechanisms = ['MCAR', 'MAR', 'MNAR']
        missing_mechanisms_datasets = {}
        for mech in missing_mechanisms:
            if mech == 'MNAR':
                data_tensor = MissingData.produce_NA(X=complete_data,p_miss=p_miss,mecha=mech,opt='logistic',p_obs=p_obs)['X_incomp']
            else:
                data_tensor = MissingData.produce_NA(X=complete_data, p_miss=p_miss, mecha=mech, p_obs=p_obs)['X_incomp']
            data_pandas = pd.DataFrame(data=data_tensor.numpy(),columns=columns)
            missing_mechanisms_datasets[mech] = data_pandas
        return missing_mechanisms_datasets

    @staticmethod
    def impute_missing(df):
        """impute missing:
                parameters:
                    df: pandas dataframe
                returns:
                    imputed_data: pandas dataframe"""
        # Filter rows that have all columns missing
        df = df.dropna(how='all')
        # Find which columns have more than 75% data missing
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': df.columns,
                                         'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        subset = list(missing_value_df.column_name[missing_value_df.percent_missing > 75])
        # Filter out the variables with more than 75% data missing
        df1 = df.drop(subset, axis=1)
        # Step 1: Use random forest imputations on the independent variables which are needed to impute the dependent
            # variables
        # You can keep all the values at default but it takes a long time to impute the missing values.
        from missingpy import MissForest
        imputer = MissForest(max_iter=10, decreasing=False, missing_values=np.nan,
                             copy=True, n_estimators=20, criterion=('mse', 'gini'),
                             max_depth=None, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features='auto',
                             max_leaf_nodes=None, min_impurity_decrease=0.0,
                             bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                             verbose=0, warm_start=False, class_weight=None)
        ##The variables selected here are only the independent ones required to impute the dependent variables
        df2 = imputer.fit_transform(df1)
        df2 = pd.DataFrame(data=df2, columns=df.columns)
        return df2
        ##The code returns a 2-D array which needs to be placed into our dataframe

def main():
    # data address to numpy and remove irrelevant information
    data_pd, data_np, columns = MissingData.data_generation(data_address)
    columns = columns[2:]
    data_pd = data_pd.drop(data_pd.columns[[0,1]],axis=1)
    data_np = np.delete(data_np,[0,1],1)
    # produce MCAR, MAR, and MNAR
    missing_mechanisms_datasets = MissingData.get_three_missing_datasets(complete_data=data_np,columns=columns)
    # deep copy missing datasets for listwise
    copied_missing_datasets = copy.deepcopy(missing_mechanisms_datasets)
    # produce three imputed missing datasets
    imputed_datasets = {}
    for mech, dataset in missing_mechanisms_datasets.items():
        imputed_datasets[mech] = MissingData.impute_missing(dataset)
    # produce three listwise deletion
    listwised_deletion = {}
    for mech, dataset in copied_missing_datasets.items():
        listwised_deletion[mech] = dataset.dropna()
    # KS test comparison
    KS_results = {}
    for mech, dataset in imputed_datasets.items():
        print(f'KS Test {mech} p-values for imputed datasets start printing-------------------------------------------')
        for column in dataset:
            p_value = ks_2samp(data_pd[column], dataset[column])[1]
            print(f"{column}'s p-value is {p_value}")
            KS_results[mech + '_' + column] = []
            KS_results[mech + '_' + column].append(p_value)
    for mech, dataset in listwised_deletion.items():
        print(f'KS Test {mech} p-values for listwise datasets start printing-------------------------------------------')
        for column in dataset:
            p_value = ks_2samp(data_pd[column], dataset[column])[1]
            print(f"{column}'s p-value is {p_value}")
            KS_results[mech + '_' + column].append(p_value)
    df = pd.DataFrame.from_dict(KS_results)
    df.to_csv('KS_results.csv', index=False)
if __name__ == '__main__':
    main()

