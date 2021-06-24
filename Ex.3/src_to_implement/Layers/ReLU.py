import numpy as np

#Standar activation function in DL
class ReLU:
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.error_tensor = None
        self.trainable = False
        self.weights = 0

    def forward(self, input_tensor):
        # f(x) = max(0,x)
        self.input_tensor = input_tensor
        # 0 if f(x) <= 0 || x if f(x) > 0
        input_tensor[input_tensor <= 0] = 0
        return np.copy(self.input_tensor)

    def backward(self, error_tensor):
        # 0 if x <= 0 || e(n) else
        # ReLU is not continuously differentiable
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor
