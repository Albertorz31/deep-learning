import numpy as np
from src_to_implement.Layers.Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input_tensor):
        self.output = np.tanh(input_tensor)
        return np.copy(self.output)

    def backward(self, error_tensor):
        gradient = 1 - self.output ** 2
        error_dw = gradient*error_tensor
        return error_dw
