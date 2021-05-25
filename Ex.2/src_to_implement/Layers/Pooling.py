import numpy as np
from scipy import signal


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape

        self.batch_size = input_tensor.shape[0]
        self.channel_size = input_tensor.shape[1]

        self.y = np.arange(0, self.input_tensor_shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0])
        self.x = np.arange(0, self.input_tensor_shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1])

        output_tensor = np.zeros([self.batch_size, self.channel_size, len(self.y), len(self.x)])

        # Create a matrix to hold maximum locations because we will do max pooling
        self.maxima_locations = np.zeros(output_tensor.shape)

        for batch_index, tensor in enumerate(input_tensor):
            for channel_index, tensor in enumerate(tensor):
                for y_index, y_prime in enumerate(self.y):
                    for x_index, x_prime in enumerate(self.x):
                        window = input_tensor[batch_index, channel_index, y_prime:y_prime + self.pooling_shape[0],
                                 x_prime:x_prime + self.pooling_shape[1]]
                        output_tensor[batch_index, channel_index, y_index, x_index] = np.amax(window)
                        self.maxima_locations[batch_index, channel_index, y_index, x_index] = np.argmax(window)

        return output_tensor

    def backward(self, error_tensor):
        # Create a zero matrix who has same shape with input tensor
        #  Because we will return this and our input tensor shape ...
        output_error_tensor = np.zeros(self.input_tensor_shape)

        for batch_index in range(self.batch_size):
            for channel_index in range(self.channel_size):
                for y_index, y_prime in enumerate(self.y):
                    for x_index, x_prime in enumerate(self.x):
                        location = int(self.maxima_locations[batch_index, channel_index, y_index, x_index])
                        y_coordination = location // self.pooling_shape[0]
                        x_coordination = location % self.pooling_shape[1]

                        output_error_tensor[batch_index, channel_index, y_prime + y_coordination,
                                            x_prime + x_coordination] += error_tensor[
                            batch_index, channel_index, y_index, x_index]

        return output_error_tensor