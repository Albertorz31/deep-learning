import numpy as np


# For L1, we are using sum of absolute values, for L2, we are focusing square!

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # calculate gradient of weigts with alpha and weights
        return self.alpha * weights

    def norm(self, weights):
        return self.alpha * np.sqrt(np.sum(weights ** 2))


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # this will calculate gradient of the weights
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # calculate norm-enhanced loss
        return self.alpha * np.sum(np.abs(weights))
