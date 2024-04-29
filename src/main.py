
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from src.data.gresearch_crypto import GResearchCryptoData
from src.trainer import get_trainer
from src.evaluation import API
from src.data.features import get_strategy

# from: https://www.kaggle.com/code/nrcjea001/lgbm-embargocv-weightedpearson-lagtarget/
def get_time_series_cross_val_splits(data, cv = 5, embargo = 3750):
    len_train = len(data['timestamp'].unique())
    len_split = len_train // cv
    test_splits = [[i * len_split, (i + 1) * len_split] for i in range(cv)]
    # fix the last test split to have all the last timestamps, in case the number of timestamps wasn't divisible by cv
    rem = len_train - len_split*cv
    if rem > 0:
        test_splits[-1] = [(cv-1)*len_split, len_train]

    train_splits = []
    for test_split in test_splits:
        test_split_min, test_split_max = test_split
        # get all of the timestamps that aren't in the test split
        if test_split_max + embargo <= len_train:
            train_split = [[0, test_split_min - embargo], [test_split_max + embargo, len_train]]
        else:
            train_split = [[0, test_split_min - embargo]]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip

def data_preprocess(data_path, n_fold, embargo):
    # Start cross validation
    df = pd.read_csv(data_path + "train.csv")
    asset_ids = df.loc[:, "Asset_ID"].unique()
    
    assets_dict = {}
    for asset_id in asset_ids:
        # get the asset from the asset_id, then reindex then to deal with the missing data by padding
        asset = df[df["Asset_ID"]==asset_id].set_index("timestamp")
        asset = asset.reindex(range(asset.index[0],asset.index[-1] + 60,60),method='pad')

        # Assert whehter the reindex is success
        assert len((asset.index[1:]-asset.index[:-1]).value_counts().index) == 1

        # the cv train test splits would be: 
        # train_splits:[[[0, 1000], [1400, 2000]], ...]
        # test_splits:[[[0, 1000]], ...]
        cv_train_test_splits = get_time_series_cross_val_splits(df, cv=n_fold, embargo=embargo)
        # Dict would be: 
        # dict[asset_id] = (asset_df, cv_train_test_splits)
        
        assets_dict[asset_id] = (asset, cv_train_test_splits)
    return assets_dict, cv_train_test_splits

def select_from_split(df, split):
    selected_index = np.concatenate([np.arange(start, end) for (start, end) in split])
    return df.iloc[split]

def train_template(assets_dict):
    # it should be understanding in following way:
    #   train_asset_id: means which assets you used to train a model
    #   test_asset_id: means which assets the model trained on train_asset_id will be used to test 
    # @TODO Now each asset used to only the asset itself
    # asset_ids = list(assets_dict.keys())
    # model_dict should be: [train_asset_id, test_asset_id, cv_train_test_split]
    model_list = [([asset_id,], [asset_id,]) for asset_id in assets_dict]
    return model_list

def main(data_path, model_name, strategy, n_fold, embargo):
    # preprocess the data from csv and make them into asset wise df and corresponding cv splits
    assets_dict, cv_splits = data_preprocess(data_path, n_fold, embargo)
    # get model from the model_name 
    #@TODO {} should be replace by hyper params if neccessary
    
    model_list = train_template(list(assets_dict.keys()))
    scores_dict = {}
    # train for each assets/shared_assets
    for cv_i, (train_split, test_split)in enumerate(cv_splits):
        trainers = {}
        for train_temp in model_list:
            #@TODO it can be modifed to multi assets features as input and predict one asset.
            # like you can make the asset_df: [asset1, asset2, asset3]
            train_asset_ids, test_asset_ids = train_temp

            train_dfs = [select_from_split(assets_dict[train_asset_id], train_split) 
                         for train_asset_id in train_asset_ids]
            # @TODO may vary the strategy for different asset
            # This step will get features and target from the original train df
            train_data = GResearchCryptoData(train_dfs, strategy)

            trainer = get_trainer(model_name, {})

            # @TODO add the params
            trainer.train(train_data, {})
            
            for test_asset_id in test_asset_ids:
                trainers[test_asset_id] = trainer
        # after training all models
        # all asset_ids for test on test splits
        # @TODO or induced from the original df
        test_df = pd.concat([select_from_split(assets_dict[asset_id], test_split) 
                         for asset_id in assets_dict]).sort_values("timestamp")
        # @TODO It can be extend to compute each asset's result
        score = test(trainers, test_df, strategy)
        scores_dict[cv_i] = score
    print("Final Results on CV:",scores_dict)

def update_history(df_test, history_test, asset_ids):
    for asset_id in asset_ids:
        asset_test = df_test[df_test["Asset_ID"] == asset_id]
        if asset_id in history_test:
            history_test[asset_id] = pd.concat([history_test[asset_id], asset_test]).set_index("timestamp")
        else:
            history_test[asset_id] = asset_test.set_index("timestamp")
    return history_test

def test(trainers, test_df, strategy):
    local_api = API(test_df)
    asset_ids = list(trainers.keys())
    strategy_func = get_strategy(strategy)
    history_test = {}
    for i, (df_test, df_pred) in enumerate(tqdm(local_api)):
        history_test = update_history(df_test, history_test, asset_ids)

        feats_test = strategy_func.get_features(history_test)
        for k in range(len(df_test.index)):
            timestamp, asset_id = df_test.iloc[k]["timestamp"], df_test.iloc[k]["Asset_ID"]
            feat_test = feats_test[history_test["Asset_ID"] == asset_id &
                                   history_test["timestamp"] == timestamp]
            y_pred = trainers[asset_id].infernce(feat_test)
            df_pred.iloc[k]["Target"] = y_pred
        local_api.predict(df_pred)
    df, score = local_api.score()
    return score


if __name__ == "__main__":
    data_path = "./data/"
    model_name = "lightgbm"
    strategy = "simple_strategy"
    n_fold = 5
    embargo = 3600 * 24 * 7
    main(data_path, model_name, strategy, n_fold, embargo)
