import numpy as np


class CrossEntropyLoss():
    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        # it should compute loss value over the batch
        # Step 1: Find the yk --> Activation/prediction from the softmax
        # We need to compute the error when it is label has 1
        # Step 2: Add epsilon to result
        # Step 3: Take ln of it
        # Step 4: Multiply with negative
        # Step 5: Sum them up --> np.sum()

        # iterate over each row (batch) and calculate the error
        # for when the label is 1

        total_error = 0
        for row_input_tensor, row_label_tensor in zip(input_tensor, label_tensor):
            # find the position of 1 in the label
            correct_label_index = np.where(row_label_tensor == 1)
            error = row_input_tensor[correct_label_index] + np.finfo(float).eps
            total_error += -np.log(error)

        self.input_tensor = input_tensor
        return (float(total_error))

    def backward(self, label_tensor):
        # Which returns error_tensor for the next layer
        #  back-propagation starts here, no need for error tensor
        # just divide label tensor by input tensor (-y/y')
        return -label_tensor / self.input_tensor
