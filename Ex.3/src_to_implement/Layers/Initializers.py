import numpy as np
import matplotlib.pyplot as plt

# Initialization is critical for non-convex optimization problems.
# different initialization strategies are required.

class Constant:

    # Constant value used typically for bias than weights
    def __init__(self, const=0.1):
        self.const = const

    def initialize(self, weights_shape, fan_in, fan_out):
        result = np.ones(weights_shape)
        result *= self.const
        return result


class UniformRandom:

    def initialize(self, weights_shape, fan_in, fan_out):

        # Draw a random number between [0,1]
        return np.random.uniform(size=weights_shape)

# fan_in: input dimension of the weights
# fan-out: output dimension of the weights

class Xavier:
    # Typically for weights, normalizes weights with respect to numbers of units
    def initialize(self, weights_shape, fan_in, fan_out):
        # Draw random values from the zero-mean gaussian
        # zero-mean = sqrt(2/ fan_out + fan-in)
        variance = np.sqrt(2 / (fan_out + fan_in))
        result = np.random.normal(0, variance, size=weights_shape)
        return result


class He:
    # Standar deviation of weights determined by size of previous layers only
    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2 / fan_in)
        result = np.random.normal(0, variance, size=weights_shape)
        return result