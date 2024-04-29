"""
The file to define and preprocess the g-research-crypto data from 
https://www.kaggle.com/competitions/g-research-crypto-forecasting/data
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any
from .features import get_strategy
import torch
from torch.utils.data import Dataset


# def select_and_reindex(df, split, asset_id):
#     asset = df[df["Asset_ID"]==asset_id].set_index("timestamp")
#     asset = asset.reindex(range(asset.index[0],asset.index[-1] + 60,60),method='pad')
    
#     # Assert whehter the reindex is success
#     assert len((asset.index[1:]-asset.index[:-1]).value_counts().index) == 1
    
#     # select index from the split: mainly used for cross-validation or train/test-split
#     # selected_index = np.concatenate([np.arange(start, end) for (start, end) in split])
#     asset = asset.loc[split]
    
#     return asset

class GResearchCryptoData(Dataset):
    """
    The dataset class for loading crypto data.
    Args:
        data_path[str]: the path to dir of te g-research crypto;
        strategy[str]: the features strategy to be chosen.
    """
    def __init__(self, dfs, target_asset_ids, strategy) -> None:
        super().__init__()
        # load data from the csv
        
        self.strategy = strategy
        self.target_asset_id = target_asset_ids
        self.load(dfs, target_asset_ids, strategy)

    def load(self, dfs, target_asset_ids=None, strategy=None):
        if strategy is None:
            strategy = self.strategy
        if target_asset_ids is None:
            target_asset_ids = self.target_asset_ids

        strategy_func = get_strategy(strategy)
        # A new feat df with the same index as the original crypto_df
        whole_feats, whole_targets = [], []
        for asset_id in dfs:
            feats = strategy_func.get_features(dfs[asset_id])
            feats.set_index(dfs[asset_id]["timestamp"])
            feats.index.names = ["date"]
            feats = feats.rename(columns={col_name:f"{col_name}_{asset_id}" for col_name in feats.columns})
            targets = dfs[asset_id].loc[:, ["Target"]]
            # drop na

            tmp_whole = pd.concat([feats, targets], axis=1)
            tmp_whole = tmp_whole.fillna(0)
            feats = tmp_whole.loc[:, feats.columns]
            targets = tmp_whole.loc[:, targets.columns]
            print("Feats after dropna", len(dfs[asset_id]), len(feats))
            whole_feats.append(feats)
            # print(dfs[asset_id])
            whole_targets.append(targets)

        self.feats = pd.concat(whole_feats, axis=1)
        # self.feats = self.feats.fillna(0.0)
        self.feats = pd.DataFrame(np.repeat(self.feats.values, len(whole_targets), axis=0), columns=self.feats.columns)
        self.targets = pd.concat(whole_targets, axis=0)

        # self.targets = self.targets.dropna()
        # self.targets = self.targets.fillna(0.0)
        # print("Feature", self.feats.iloc[50:60])
        # sss

    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, index) -> Any:
        return self.feats.iloc[index], self.targets.iloc[index]


# class GResearchCryptoData(Dataset):
#     """
#     The dataset class for loading crypto data.
#     Args:
#         data_path[str]: the path to dir of te g-research crypto;
#         split[list]: a list containing split start and end, e.g., [[0.1, 0.4], [0.5,0.9]...];
#         asset_id[int]: the id indicates which asset to be select as the data;
#         strategy[str]: the features strategy to be chosen.
#     """
#     def __init__(self, df, strategy) -> None:
#         super().__init__()
#         # load data from the csv
#         self.asset_id = asset_id
#         self.strategy = strategy
#         self.split = split
#         self.load(df, split, asset_id)

#     def load(self, crypto_df, split, asset_id=None, strategy=None):

#         if asset_id is None:
#             asset_id = self.asset_id
#         if strategy is None:
#             strategy = self.strategy
#         # Deal with missing data
#         asset = select_and_reindex(crypto_df, split, asset_id)

#         strategy = features.get_strategy(strategy)

#         # A new feat df with the same index as the original crypto_df
#         self.feats = strategy.get_features(asset)
#         self.targets = asset["Target"]

#     def __len__(self):
#         return len(self.feats)
    
#     def __getitem__(self, index) -> Any:
#         return self.feats.iloc[index], self.targets.iloc[index]
    





    





    