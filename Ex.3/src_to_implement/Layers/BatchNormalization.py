from Helpers import *
from TanH import *


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()

        self.trainable = True
        self.channels = channels

        self.mean = None
        self.variance = None
        self.test_mean = None
        self.test_variance = None

        self.bias = np.zeros(channels)
        self.weights = np.ones(channels)

        self.gradient_bias = None
        self.gradient_weights = None

        self.optimizer = None  # optimizer
        self.bias_optimizer = None

        self.weights_initializer = None
        self.bias_initializer = None

        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            tensor = np.transpose(tensor, [0, 2, 3, 1])
            # reshape
            new_tensor = np.reshape(tensor, (tensor.shape[0] * tensor.shape[1] * tensor.shape[2],
                                             tensor.shape[3]), order="C")
            return new_tensor

        if len(tensor.shape) == 2:
            tensor = tensor.reshape(self.input_tensor.shape[0], int(tensor.shape[0] / self.input_tensor.shape[0]),
                                    self.channels)
            # B,MN,H transpose B,H,MN
            # B,H,MN reshape B,M,N,H
            new_tensor = np.transpose(tensor, (0, 2, 1)).reshape(self.input_tensor.shape)

            return new_tensor

    def forward(self, input_tensor, alpha=0.8):
        epsilon = np.finfo(float).eps
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 2:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

            if not self.testing_phase:
                # mini-batch mean
                new_mean = np.mean(input_tensor, axis=0)
                # mini-batch variance
                new_variance = np.var(input_tensor, axis=0)

                self.test_mean = alpha * self.mean + (1 - alpha) * new_mean
                self.test_variance = alpha * self.variance + (1 - alpha) * new_variance

                self.mean = new_mean
                self.variance = new_variance

                # NORMALIZE
                X_hat = (input_tensor - self.mean) / np.sqrt(self.variance + epsilon)

            # NORMALIZE test time
            else:
                X_hat = (input_tensor - self.test_mean) / np.sqrt(self.test_variance + epsilon)

            self.X_hat = X_hat
            out = self.weights * X_hat + self.bias

        elif len(input_tensor.shape) == 4:
            # extract the dimensions
            B, H, M, N = input_tensor.shape
            self.mean = np.mean(input_tensor, axis=(0, 2, 3))
            self.variance = np.var(input_tensor, axis=(0, 2, 3))

            if not self.testing_phase:
                # mini-batch mean
                new_mean = np.mean(input_tensor, axis=(0, 2, 3))
                # mini-batch variance
                new_variance = np.var(input_tensor, axis=(0, 2, 3))

                '''Moving average calculations for test time'''
                self.test_mean = alpha * self.mean.reshape((1, H, 1, 1)) + (1 - alpha) * new_mean.reshape((1, H, 1, 1))
                self.test_variance = alpha * self.variance.reshape((1, H, 1, 1)) + (1 - alpha) * new_variance.reshape(
                    (1, H, 1, 1))

                self.mean = new_mean
                self.variance = new_variance

                # NORMALIZE
                X_hat = (input_tensor - self.mean.reshape((1, H, 1, 1))) / np.sqrt(
                    self.variance.reshape((1, H, 1, 1)) + epsilon)

            # NORMALIZE test time
            else:
                X_hat = (input_tensor - self.test_mean.reshape((1, H, 1, 1))) / np.sqrt(
                    self.test_variance.reshape((1, H, 1, 1)) + epsilon)

            self.X_hat = X_hat
            out = self.weights.reshape((1, H, 1, 1)) * X_hat + self.bias.reshape((1, H, 1, 1))

        return out

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            out = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor),
                                       self.weights, self.mean, self.variance, 1e-15)
            out = self.reformat(out)
            self.gradient_weights = np.sum(error_tensor * self.X_hat, axis=(0, 2, 3))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        elif len(error_tensor.shape) == 2:
            out = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance, 1e-15)
            self.gradient_weights = np.sum(error_tensor * self.X_hat, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

        '''Update with optimizers'''
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return out

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

