import numpy as np

#Is a Activation function
class SoftMax:
    def __init__(self):
        self.input_tensor = None
        self.error_tensor = None
        self.y = None
        self.trainable = False
        self.weights = 0

    def forward(self, input_tensor):
        # If your input consists of several samples, he takes a 1-dimensional input and
        # then he takes a 2-dimensional input

        # y = exp(x_k - x_max)/ sum(exp(x_k)
        # a = exp(X_k)
        s = np.max(input_tensor, axis=1)
        # np.newaxis is used to increase the dimension of the existing array by one more dimension
        s = s[:, np.newaxis]
        exp_input = np.exp(input_tensor - s)
        # b = sum(exp(X_k)
        sumatory = np.sum(exp_input,axis=1)
        sumatory = sumatory[:, np.newaxis]
        # y = a/b
        self.y = exp_input / sumatory
        # reference: https://stackoverflow.com/a/39558290
        return exp_input / sumatory

    def backward(self, error_tensor):
        # Compute for every element of the batch
        # e = sum(E_n*y)
        error_with_input = np.sum(error_tensor*self.y, axis=1, keepdims=True)
        # result = y* ( E - e)
        result = self.y * (error_tensor - error_with_input)
        return result
