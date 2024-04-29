import numpy as np
import lightgbm as lgb
from .trainer import Trainer
from sklearn.linear_model import ElasticNet

class ElasticNetTrainer(Trainer):
    def __init__(self, h_params) -> None:
        params = {
            
         }
        params.update(h_params)
        self.params = params

    def train(self, data, params=None, valid_data=None):
        if params is None:
            params = self.params
        else:
            params_copy = self.params.copy()
            params_copy.update(params)
            params = params_copy
        X, y = data.feats.to_numpy(), data.targets.to_numpy()
        self.reg = ElasticNet(alpha=0.001, l1_ratio=0.5).fit(X, y)
        
    def inference(self, x_input):
        return self.reg.predict(x_input)
    