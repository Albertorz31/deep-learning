import numpy as np
import matplotlib.pyplot as plt


class Constant:
    def __init__(self, const=0.1):
        self.const = const

    def initialize(self, weights_shape, fan_in, fan_out):
        result = np.ones(weights_shape)
        result *= self.const
        return result


class UniformRandom:

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)


class Xavier:

    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2 / (fan_out + fan_in))
        result = np.random.normal(0, variance, size=weights_shape)
        return result


class He:

    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2 / fan_in)
        result = np.random.normal(0, variance, size=weights_shape)
        return result