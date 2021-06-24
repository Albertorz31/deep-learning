import numpy as np
from Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 / (1 + np.exp(-1*input_tensor))
        return self.activation

    def backward(self, error_tensor):
        error_tensor = self.activation * (1 - self.activation) * error_tensor
        return error_tensor
