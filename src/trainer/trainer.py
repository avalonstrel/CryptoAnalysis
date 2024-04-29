
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, h_params) -> None:
        pass
    
    @abstractmethod
    def train(self, data, params):
        pass

    @abstractmethod
    def inference(self, x_input):
        pass
