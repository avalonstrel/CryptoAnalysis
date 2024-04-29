import numpy as np
import lightgbm as lgb
from .trainer import Trainer

def correlation(a, train_data):
    b = train_data.get_label()
    
    a = np.ravel(a)
    b = np.ravel(b)

    len_data = len(a)
    mean_a = np.sum(a) / len_data
    mean_b = np.sum(b) / len_data
    var_a = np.sum(np.square(a - mean_a)) / len_data
    var_b = np.sum(np.square(b - mean_b)) / len_data

    cov = np.sum((a * b))/len_data - mean_a*mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return 'corr', corr, True


class LightGBMTrainer(Trainer):
    def __init__(self, h_params) -> None:
        params = {'num_leaves': 31, 'objective': 'regression'}
        params['metric'] = 'rmse'
        seed0 = 17
        params = {
            'early_stopping_rounds': 50,
            'objective': 'regression',
            'metric': 'rmse',
        #     'metric': 'None',
            'boosting_type': 'gbdt',
            'max_depth': 5,
            'verbose': -1,
            'max_bin':600,
            'min_data_in_leaf':50,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'subsample_freq': 1,
            'feature_fraction': 1,
            'lambda_l1': 0.5,
            'lambda_l2': 2,
            'seed':seed0,
            'feature_fraction_seed': seed0,
            'bagging_fraction_seed': seed0,
            'drop_seed': seed0,
            'data_random_seed': seed0,
            'extra_trees': True,
            'extra_seed': seed0,
            'zero_as_missing': True,
            "first_metric_only": True
         }
        params.update(h_params)
        self.params = params

    def train(self, data, params=None, valid_data=None):
        num_round = 10
        if params is None:
            params = self.params
        else:
            params_copy = self.params.copy()
            params_copy.update(params)
            params = params_copy
        
        train_data = lgb.Dataset(data.feats.to_numpy(), label=data.targets.to_numpy())
        val_data = lgb.Dataset(data.feats.to_numpy()[-10000:], label=data.targets.to_numpy()[-10000:])
        self.bst = lgb.train(params, train_data, feval = correlation, valid_sets=[val_data])

    def inference(self, x_input):
        return self.bst.predict(x_input)
    