import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input = None

    def forward(self, input_tensor, label_tensor):
        self.input = input_tensor.copy()
        eps = np.finfo(float).eps
        return np.sum(np.where(label_tensor == 1, -np.log(input_tensor + eps), 0))

    def backward(self, label_tensor):
        return -label_tensor / self.input
