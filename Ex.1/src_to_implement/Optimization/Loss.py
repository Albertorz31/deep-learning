import numpy as np


# It is used to optimize classification models.
# The understanding of the cross entropy is based on the understanding of the activation function of Softmax.
class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None

    # The purpose of the cross entropy is to take the output probabilities (P) and measure
    # the distance from the truth values
    def forward(self, input_tensor, label_tensor):

        n = len(label_tensor)
        self.input_tensor = input_tensor
        # Binary Cross Entropy Loss
        loss = np.sum(-label_tensor * np.log(input_tensor+np.finfo(float).eps))
        return np.squeeze(loss)



    def backward(self, label_tensor):
        # Which returns error_tensor for the next layer
        # back-propagation starts here, no need for error tensor
        # just divide label tensor by input tensor (-y/y')
        return -(label_tensor / self.input_tensor)
