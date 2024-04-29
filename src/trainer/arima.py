import numpy as np
from .trainer import Trainer
from statsmodels.tsa.arima.model import ARIMA


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

        model = ARIMA(data, order=(1, 1, 1))  # Example: ARIMA(1,1,1)
        self.fited = model.fit()
   
    def inference(self, x_input):
        return self.reg.forecast(x_input)
    