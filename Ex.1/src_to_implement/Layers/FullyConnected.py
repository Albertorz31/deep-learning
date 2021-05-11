import numpy as np
import sys

sys.path.insert(0, './Optimization')
import Optimizers


class FullyConnected():
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.input_tensor = None
        self.output_tensor = None
        self.error_tensor = None

        # adding bias self.input_size + 1
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self._optimizer = None
        self.gradient_weights = None

    def forward(self, input_tensor):
        # [x1 x2 x3 . . . 1]
        self.input_tensor = np.append(input_tensor, np.ones((len(input_tensor), 1)), axis=1)
        # X'* W' = Y_hat' (predicted value)
        self.output_tensor = np.matmul(self.input_tensor, self.weights)

        return np.copy(self.output_tensor)

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        # E(n-1)' = E(n)'*W'transpose
        previous_error = np.matmul(error_tensor, np.transpose(self.weights))
        # we removing bias part
        previous_error = previous_error[:, :-1]

        # Gradient = X'transpose*E(n)'
        self.gradient_weights = np.matmul(np.transpose(self.input_tensor), self.error_tensor)

        if self._optimizer:
            # Optimizers.Sgd(w, g).calculate_update
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return np.copy(previous_error)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        # _optimizer = Optimizers.Sgd(value)
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.gradient_weights = value
