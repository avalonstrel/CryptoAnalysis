import json
from tabulate import tabulate
import pandas as pd
import numpy as np

methods = ["linearregression", "lightgbm", "elasticnet", "iInformer", "iReformer", 'iTransformer']
asset_ids = ["0", "1", "3"]
data_path = "data/"
weight_df = pd.read_csv(data_path + "asset_details.csv")
id2weight = {weight_df.loc[ind, "Asset_ID"]:weight_df.loc[ind, "Weight"] for ind in weight_df.index}
id2names = {str(weight_df.loc[ind, "Asset_ID"]):weight_df.loc[ind, "Asset_Name"] for ind in weight_df.index}
strategy = "simple_strategy"
n_fold = 6
embargo = 3600 * 24 * 1

def methods_table():
    table_data = [["Method",] + [id2names[asset_id] for asset_id in asset_ids] + ["Weighted"]]

    for method in methods:
        row_data = [method]
        save_name = f"exprs/{method}_{strategy}_nfold{n_fold}_emb{embargo}.json"
        result = json.load(open(save_name, 'r'))
        # print(result)
        for asset_id in asset_ids: 

            avg_asset = np.mean([result[1][key][asset_id] for key in result[1]])
            row_data.append(avg_asset)
        row_data.append(np.mean(list(result[0].values())))
        table_data.append(row_data)

    table = tabulate(table_data, headers="firstrow", tablefmt="latex")
    print(table)


def deep_methods_table():

    methods = ["iInformer", "iReformer", 'iTransformer']
    table_data = [["Method",] + [id2names[asset_id] for asset_id in asset_ids] + ["Weighted"]]

    for method in methods:
        row_data = [method]
        save_name = f"exprs/{method}_{strategy}_nfold{n_fold}_emb{embargo}.json"
        result = json.load(open(save_name, 'r'))
        # print(result)
        for asset_id in asset_ids: 

            avg_asset = np.mean([result[1][key][asset_id] for key in result[1]])
            row_data.append(avg_asset)
        row_data.append(np.mean(list(result[0].values())))
        table_data.append(row_data)

    table = tabulate(table_data, headers="firstrow", tablefmt="latex")
    print(table)


def features_table():
    table_data = [["Strategy",] + [id2names[asset_id] for asset_id in asset_ids] + ["Weighted"]]
    methods = ["lightgbm"]
    strategies = ["simple_strategy", "clean_strategy", "cleani_strategy"]
    for method in methods:
        for strategy in strategies:
            row_data = [f"{strategy}"]
            save_name = f"exprs/{method}_{strategy}_nfold{n_fold}_emb{embargo}.json"
            result = json.load(open(save_name, 'r'))
            # print(result)
            for asset_id in asset_ids: 
                avg_asset = np.mean([result[1][key][asset_id] for key in result[1]])
                row_data.append(avg_asset)
            row_data.append(np.mean(list(result[0].values())))
            table_data.append(row_data)

    table = tabulate(table_data, headers="firstrow", tablefmt="latex")
    print(table)


deep_methods_table()
