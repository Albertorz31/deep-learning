import numpy as np
from Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        error_tensor = (1 - np.square(self.activation)) * error_tensor
        return error_tensor
