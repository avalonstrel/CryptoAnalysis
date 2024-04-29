
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
import os
import pickle as pkl
from src.data.gresearch_crypto import GResearchCryptoData
from src.trainer import get_trainer

from src.evaluation import API, weighted_correlation
from src.data.features import get_strategy

# from: https://www.kaggle.com/code/nrcjea001/lgbm-embargocv-weightedpearson-lagtarget/
def get_time_series_cross_val_splits(data, cv = 5, embargo = 3750):
    len_train = len(data['timestamp'].unique())
    aug_num = 3
    len_split = len_train // (cv + aug_num)
    test_splits = [[[i * len_split, (i + 1) * len_split]] 
                        for i in range(cv + aug_num) 
                            if (i * len_split) >= (2 * len_split + embargo) and ((i+1) * len_split <= len_train)]
    # fix the last test split to have all the last timestamps, in case the number of timestamps wasn't divisible by cv
    # rem = len_train - len_split*cv
    # if rem > 0:
    #     test_splits[-1] = [[(cv-1)*len_split, len_train]]

    train_splits = []
    for test_split in test_splits:
        test_split_min, test_split_max = test_split[0]
        # get all of the timestamps that aren't in the test split
        train_split = []
        # if test_split_max + embargo <= len_train:
        #     train_split.append([test_split_max + embargo, len_train])
        if test_split_min - embargo > 0:
            train_split.append([0, test_split_min - embargo])
            
        train_splits.append(train_split)
    # print(train_splits, test_splits)
    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip

def data_preprocess(data_path, n_fold, embargo):
    weight_df = pd.read_csv(data_path + "asset_details.csv")
    id2weight = {weight_df.loc[ind, "Asset_ID"]:weight_df.loc[ind, "Weight"] for ind in weight_df.index}

    # Start cross validation
    df = pd.read_csv(data_path + "train.csv")
    asset_ids = df.loc[:, "Asset_ID"].unique()
    assets_dict = {}
    # get min/max timestamp
    min_timestamp, max_timestamp = df["timestamp"].min(), df["timestamp"].max()

    for asset_id in asset_ids:
        # get the asset from the asset_id, then reindex then to deal with the missing data by padding
        asset = df[df["Asset_ID"]==asset_id].set_index("timestamp")
        print("Asset info:", asset.index[0], asset.index[-1], "Genral info:", min_timestamp, max_timestamp)
        asset = asset.reindex(range(min_timestamp, max_timestamp + 60, 60),method='pad')
        asset = asset.fillna(0)
        
        asset.index.names = ["date"]
        asset.loc[:, "timestamp"] = asset.index
        # Assert whehter the reindex is success
        assert len((asset.index[1:]-asset.index[:-1]).value_counts().index) == 1
        
        # the cv train test splits would be: 
        # train_splits:[[[0, 1000], [1400, 2000]], ...]
        # test_splits:[[[0, 1000]], ...]
        cv_train_test_splits = get_time_series_cross_val_splits(df, cv=n_fold, embargo=embargo)
        # Dict would be: 
        # dict[asset_id] = (asset_df, cv_train_test_splits)
        
        assets_dict[asset_id] = (asset, cv_train_test_splits)
    return assets_dict, cv_train_test_splits, id2weight

def select_from_split(df, split):
    selected_index = np.concatenate([np.arange(start, end) for (start, end) in split])
    return df.iloc[selected_index]

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
    save_name = f"exprs/{model_name}_{strategy}_nfold{n_fold}_emb{embargo}/results.json"
    save_path = f"exprs/{model_name}_{strategy}_nfold{n_fold}_emb{embargo}"
    os.makedirs(save_path, exist_ok=True)
    # preprocess the data from csv and make them into asset wise df and corresponding cv splits
    assets_dict, cv_splits, id2weight = data_preprocess(data_path, n_fold, embargo)
    # get model from the model_name 
    #@TODO {} should be replace by hyper params if neccessary
    
    model_list = train_template(list(assets_dict.keys()))
    scores_dict = {}
    whole_scores_dict = {}
    # train for each assets/shared_assets
    for cv_i, (train_split, test_split) in enumerate(cv_splits):
        trainers = {}
        for train_temp in model_list:
            #@TODO it can be modifed to multi assets features as input and predict one asset.
            # like you can make the asset_df: [asset1, asset2, asset3]
            train_asset_ids, test_asset_ids = train_temp

            train_dfs = {train_asset_id:select_from_split(assets_dict[train_asset_id][0], train_split) 
                         for train_asset_id in train_asset_ids}
            # @TODO may vary the strategy for different asset
            # This step will get features and target from the original train df
            train_data = GResearchCryptoData(train_dfs, test_asset_ids, strategy)

            trainer = get_trainer(model_name, {})

            # @TODO add the params
            trainer.train(train_data, {})
            
            for test_asset_id in test_asset_ids:
                trainers[test_asset_id] = trainer
        # after training all models
        # all asset_ids for test on test splits
        # @TODO or induced from the original df
        test_df = pd.concat([select_from_split(assets_dict[asset_id][0], test_split) 
                         for asset_id in assets_dict]).sort_values("timestamp")
        # print(test_df)
        # @TODO It can be extend to compute each asset's result
        score, in_scores_dict = test(trainers, test_df, strategy, id2weight, save_path)
        scores_dict[cv_i] = score
        whole_scores_dict[cv_i] = in_scores_dict
        json.dump([scores_dict, whole_scores_dict], open(save_name, 'w'))
    print("Final Results on CV:",scores_dict)

def update_history(df_test, history_test, asset_ids):
    for asset_id in asset_ids:
        asset_test = df_test[df_test["Asset_ID"] == asset_id].set_index("timestamp")
        asset_test.index.names = ["date"]
        asset_test.loc[:, "timestamp"] = asset_test.index
        if asset_id in history_test:
            history_test[asset_id] = pd.concat([history_test[asset_id], asset_test])
        else:
            history_test[asset_id] = asset_test
        history_test[asset_id] = history_test[asset_id].dropna()
    return history_test

def save_model_rsults(trainers, preds, targets, save_path):
    from src.trainer.lightgbm import LightGBMTrainer
    for asset_id in trainers:
        trainer = trainers[asset_id]
        if isinstance(trainer, LightGBMTrainer):
            bst = trainer.bst
        bst.save_model(os.path.join(save_path, f"model{asset_id}.txt"))
    # pkl.dump()
    pkl.dump({"targets":targets, "preds":preds}, open(os.path.join(save_path, "preds_targets.pkl"), 'wb'))


def test(trainers, test_df, strategy, id2weight, save_path):
    asset_ids = list(trainers.keys())
    strategy_func = get_strategy(strategy)
    history_test = update_history(test_df, {}, asset_ids)
    # feats_test = {asset_id:strategy_func.get_features(history_test[asset_id]) for asset_id in asset_ids}
    # targets_test = {asset_id:history_test[asset_id].loc[:, "Target"] for asset_id in asset_ids}
    whole_preds, whole_targets, whole_weights = [], [], []
    scores_dict = {}
    for asset_id in tqdm(asset_ids):
        if len(history_test[asset_id]) == 0:
            continue
        feat_test, target_test = strategy_func.get_features(history_test[asset_id]), history_test[asset_id].loc[:, ["Target"]]

        tmp_whole = pd.concat([feat_test, target_test], axis=1)
        tmp_whole = tmp_whole.dropna()
        feat_test = tmp_whole.loc[:, feat_test.columns]
        target_test = tmp_whole.loc[:, target_test.columns]
        print("Test", asset_id, len(feat_test), len(target_test))
        if len(feat_test) == 0:
            continue
        if 'former' in model_name:
            y_pred, target_test = trainers[asset_id].inference([feat_test, target_test])
        else:
            y_pred = trainers[asset_id].inference(feat_test)
            target_test = target_test.to_numpy()
        whole_preds.append(y_pred)
        whole_targets.append(target_test)
        whole_weights.append(np.repeat(id2weight[asset_id], len(y_pred)))
        scores_dict[int(asset_id)] = weighted_correlation(whole_preds[-1], whole_targets[-1], whole_weights[-1])
        if int(asset_id) == 1:
            print(feat_test.columns)
            save_model_rsults(trainers, whole_preds[-1], whole_targets[-1], save_path)
            sss
    whole_preds = np.concatenate(whole_preds, axis=0)
    whole_targets = np.concatenate(whole_targets, axis=0)
    whole_weights = np.concatenate(whole_weights, axis=0)
    # print(whole_preds, whole_targets)
    score = weighted_correlation(whole_preds, whole_targets, whole_weights)
    
    print("Score inside CV:", score, scores_dict)
    # Do not use the test api
    # local_api = API(test_df)
    # for i, (df_test, df_pred) in enumerate(tqdm(local_api)):
    #     history_test = update_history(df_test, history_test, asset_ids)
    #     feats_test = {asset_id:strategy_func.get_features(history_test[asset_id]) for asset_id in history_test}
    #     for k in range(len(df_test.index)):
    #         timestamp, asset_id = df_test.iloc[k]["timestamp"], df_test.iloc[k]["Asset_ID"]
    #         feat_test = feats_test[asset_id][history_test[asset_id]["timestamp"] == timestamp]
    #         y_pred = trainers[asset_id].inference(feat_test)
    #         df_pred.loc[df_pred.index[k], "Target"] = y_pred[0]
    #     local_api.predict(df_pred)
    #     local_api.predict(df_pred)
    # df, score = local_api.score()
    return score, scores_dict


if __name__ == "__main__":
    data_path = "./data/"
    model_name =  "lightgbm" ## "lightgbm"  "lightgbm"
    strategy = "simple_strategy"
    n_fold = 6
    embargo = 3600 * 24 * 1
    main(data_path, model_name, strategy, n_fold, embargo)
    # for strategy in ["clean_strategy",]:
    #     for model_name in ["linearregression", "lightgbm", "elasticnet",]:
    #         main(data_path, model_name, strategy, n_fold, embargo)
    # for strategy in ["simple_strategy", "clean_strategy", "cleani_strategy"]:
    #     for model_name in ["iInformer", "iReformer", "iTransformer"]:
    #         main(data_path, model_name, strategy, n_fold, embargo)

     
