import numpy as np
from src_to_implement.Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, prob):
        super().__init__()
        self.probability = prob

    def forward(self, input_tensor):
        if not self.testing_phase:
            # ! We are in the training stage
            # we should set some of them into zero
            # probability represent "keep probability." so we will have that much 1
            # Firstly create a mask with the binomial random values, to have 0 and 1.
            # Shape should be same with the input tensor
            self.mask = np.random.binomial(1, self.probability, input_tensor.shape)
            # then divide into self.probability
            # But, we can "fix" average value for each neuron in the network by multiplying it by 1/p during training time.
            self.mask = self.mask / self.probability
            # print ("train in dropout")
        elif self.testing_phase:
            # ! We are in the test stage.
            # we should create mask with the 1 it is same with shape of input tensor
            self.mask = np.ones(input_tensor.shape)
            # print ("test in dropout")

        # at the end, multiply input tensor with the mask
        return self.mask * input_tensor

    def backward(self, error_tensor):
        return error_tensor * self.mask
