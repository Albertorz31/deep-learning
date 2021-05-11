import numpy as np


class SoftMax:
    def __init__(self):
        self.input_tensor = None
        self.error_tensor = None
        self.y = None
        self.trainable = False

    def forward(self, input_tensor):
        # a = exp(X_k)
        exp_input = np.exp(input_tensor - np.max(input_tensor))
        # y = a/b
        self.y = exp_input / exp_input.sum(axis=0) # b = sum(exp(X_k)
        return self.y

    def backward(self, error_tensor):
        # Compute for every element of the batch
        # e = sum(E_n*y)
        error_with_input = np.sum(error_tensor*self.y, axis=1, keepdims=True)
        # result = y* ( E - e)
        result = self.y * (error_tensor - error_with_input)
        return result
