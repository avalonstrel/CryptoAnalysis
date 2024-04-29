"""
Features for the price data. Currently it will be the combinantion of the TA-Lib results.
Just wrap the talib functions.
"""

import numpy as np
import talib
from talib import abstract
import pandas as pd
from sklearn.preprocessing import StandardScaler

def _adj_column_names(ts):
    """
    ta-lib expects columns to be lower case; to be consistent,
    change date index
    """
    # print(ts)
    ts.columns = [col.lower().replace(' ','_') for col in ts.columns]
    ts.index.names = ['date']
    return ts

def _new_feat_name(ind_name, params, output_name):
    param_tag = "_".join([f"{k}-{v}" for k, v in params.items()])
    return f"{ind_name}_{param_tag}_{output_name}"



simple_strategy = {
    # Moving average
    "SMA":{"timeperiod":15},
    "SMA":{"timeperiod":60},
    "EMA":{},
    # Momentum
    "ADX":{"timeperiod":30},
    "ADX":{"timeperiod":60},
    "BOP":{},
    "ADXR":{},
    "APO":{},
    "BOP":{},
    "CCI":{},
    "CMO":{},
    "PPO":{},
    "MOM":{},
    "LCM":{},
    "LRM":{},
    "LCM":{"timeperiod":30},
    "LRM":{"timeperiod":30},
    "LCM":{"timeperiod":60},
    "LRM":{"timeperiod":60},
    "ROC":{},
    "ROCP":{},
    "ROCR":{},
    "ATR":{},
}

clean_strategy = {
    # Moving average
    "SMA":{"timeperiod":15},
    "SMA":{"timeperiod":60},
    "EMA":{"timeperiod":30},
    "EMA":{"timeperiod":60},
    # Momentum
    "ADX":{"timeperiod":30},
    "ADX":{"timeperiod":60},
    "BOP":{},
    "ADXR":{},
    "APO":{},
    "BOP":{},
    "CCI":{},
    "CMO":{},
    "PPO":{},
    "MOM":{},
    "ROC":{},
    "ROCP":{},
    "ROCR":{},
    "ATR":{},
}

cleani_strategy = {
    # Moving average
    "SMA":{"timeperiod":15},
    "SMA":{"timeperiod":60},
    "EMA":{"timeperiod":30},
    "EMA":{"timeperiod":60},
    # Momentum
    "ADX":{"timeperiod":30},
    "ADX":{"timeperiod":60},
    "BOP":{},
    "ADXR":{},
    "APO":{},
    "BOP":{},
    "CCI":{},
    "CMO":{},
    "PPO":{},
    "MOM":{},
    "ROC":{},
    "ROCP":{},
    "ROCR":{},
    "ATR":{},
    "ICLOSE":{},
    "IOPEN":{},
    "IVOLUME":{}
}

def features(output_names):
    def wrapper(func):
        func.output_names = output_names
        return func
    return wrapper
# # LCM:log_close_mean
@features(["real"])
def LCM(df, timeperiod=14):
    lcm = np.log( np.array(df[f"close"]) /  np.roll(np.append(np.convolve( np.array(df[f"close"]), np.ones(timeperiod)/timeperiod, mode="valid"), np.ones(timeperiod-1)), timeperiod-1))
    lcm[np.isinf(lcm)] = np.nan
    return lcm

# # LRM: log_return_mean
@features(["real"])
def LRM(df, timeperiod=14):
    lrm = np.log( np.array(df[f"close"]) /  np.roll(np.array(df[f"close"]), timeperiod))
    lrm[np.isinf(lrm)] = np.nan
    return lrm

@features(["real"])
def ICLOSE(df):
    return np.array(df["close"])

@features(["real"])
def IVOLUME(df):
    return np.array(df["volume"])

@features(["real"])
def IOPEN(df):
    return np.array(df["open"])


class Strategy:
    """
    Construct features accroding to indicators and parameters
    e.g.,
    {
        "SMA":{"timeperiod":10,},
        ...
    }
    """
    def __init__(self, indicators) -> None:
        self.indicators = indicators

    def get_features(self, df):
        df = _adj_column_names(df.copy(deep=True))
        feat_df = pd.DataFrame(index=df.index)
        for indicator in self.indicators:
            if hasattr(abstract, indicator):
                ind_func = getattr(abstract, indicator)
            elif indicator in globals():
                ind_func = globals()[indicator]
            else:
                raise NotImplementedError
            output_names = ind_func.output_names
            outputs = ind_func(df, **self.indicators[indicator])
            
            if len(output_names) == 1:
                outputs = [outputs]
            for output_name, output in zip(output_names, outputs):
                new_feat_name = _new_feat_name(indicator, 
                                               self.indicators[indicator], 
                                               output_name)
                feat_df.loc[:, new_feat_name] = output
        
        # feat_df = feat_df.fillna(0.0)
        # feat_df = feat_df.dropna()
        # print(feat_df, feat_df.isna())
        # if normalize
        # feat_df = self.normalize(feat_df)
        return feat_df
    
    def normalize(self, df):
        
        # simple preprocessing of the data 
        scaler = StandardScaler()
        normalized_np = scaler.fit_transform(df.to_numpy())
        return pd.DataFrame(normalized_np, columns=df.columns, index=df.index)

def get_strategy(strategy_name):
    # print(globals())
    return Strategy(globals()[strategy_name])