import numpy as np
from src_to_implement.Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input_tensor):
        self.output = 1/(1 + np.exp(-input_tensor))
        return np.copy(self.output)

    def backward(self, error_tensor):
        gradient = self.output*(1 - self.output)
        error_dw = gradient*error_tensor
        return error_dw