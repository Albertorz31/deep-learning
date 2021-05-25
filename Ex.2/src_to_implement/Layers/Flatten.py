import numpy as np


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = np.shape(input_tensor)  # batch, channel, spatial1, spatial2
        output_shape = np.array([self.input_shape[0], np.prod(self.input_shape[1:])])
        output = np.reshape(input_tensor, output_shape)

        return np.copy(output)

    def backward(self, error_tensor):
        output = np.reshape(error_tensor, self.input_shape)

        return np.copy(output)