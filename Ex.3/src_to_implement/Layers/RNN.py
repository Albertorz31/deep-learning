import numpy as np
import copy
from Base import BaseLayer
from FullyConnected import FullyConnected
from Sigmoid import Sigmoid
from TanH import TanH


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = None
        self.prev_hidden_state = None
        self._memorize = False
        self.FC_2 = FullyConnected(hidden_size, output_size)
        self.FC_1 = FullyConnected(hidden_size + input_size, hidden_size)
        self.sigmoid = Sigmoid()
        self.tanH = TanH()
        self.error_tensor_h = None
        self._optimizer = None

    def forward(self, input_tensor):  ##(batch_size, input_size)
        batch_size = input_tensor.shape[0]
        new_input = np.zeros(
            (1, self.input_size + self.hidden_size))  ## need to satisfy the forward parameter requirement
        self.hidden_state = np.zeros((batch_size + 1, self.hidden_size))
        self.input_tensor = input_tensor

        if self.memorize == False:
            self.hidden_state = np.zeros((batch_size + 1, self.hidden_size))
        else:
            self.hidden_state[0] = self.prev_hidden_state

        output = np.zeros((batch_size, self.output_size))
        self.FC_1_input_tensor = []
        self.FC_2_input_tensor = []
        self.TanH_output = np.zeros((batch_size, self.hidden_size))
        self.Sigmoid_output = np.zeros((batch_size, self.output_size))
        for i in range(batch_size):
            new_input[0, 0:self.input_size] = input_tensor[i]
            new_input[0, self.input_size:] = self.hidden_state[i]
            self.hidden_state[i + 1] = self.tanH.forward(self.FC_1.forward(new_input))
            self.FC_1_input_tensor.append(self.FC_1.input_tensor)
            output[i] = self.sigmoid.forward(self.FC_2.forward(np.expand_dims(self.hidden_state[i + 1], axis=0)))[0]
            self.FC_2_input_tensor.append(self.FC_2.input_tensor)

            self.TanH_output[i] = self.tanH.activation
            self.Sigmoid_output[i] = self.sigmoid.activation
        self.prev_hidden_state = self.hidden_state[-1]
        return output

    def backward(self, error_tensor):

        batch_size = error_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.input_size))
        self.gradient_weights_FC2 = 0
        self.gradient_weights_FC1 = 0

        self.error_tensor_h = np.zeros(self.hidden_size)

        for i in reversed(range(batch_size)):
            self.FC_2.input_tensor = self.FC_2_input_tensor[i]
            self.FC_1.input_tensor= self.FC_1_input_tensor[i]
            self.tanH.activation = self.TanH_output[i].reshape(1, -1)
            self.sigmoid.activation = self.Sigmoid_output[i].reshape(1, -1)

            error_tensor_FC2 = self.FC_2.backward(self.sigmoid.backward(np.expand_dims(error_tensor[i], axis=0)))

            grad_vor_tanh = error_tensor_FC2 + self.error_tensor_h
            error_tensor_FC1 = self.FC_1.backward(self.tanH.backward(grad_vor_tanh))
            error_tensor_x = error_tensor_FC1[:, :self.input_size]
            self.error_tensor_h = error_tensor_FC1[:, self.input_size:]
            output_tensor[i] = error_tensor_x
            self.gradient_weights_FC2 += self.FC_2.gradient_weights
            self.gradient_weights_FC1 += self.FC_1.gradient_weights
        self.weights_FC2 =self.FC_2.weights
        self.weight = self.FC_1.weights
        #self.gradient_weights = self.gradient_weights_FC1

        if self.optimizer is not None:
            self.weights_FC2 = self.optimizer2.calculate_update(self.weights_FC2, self.gradient_weights_FC2)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights_FC1)

        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self.optimizer2 = optimizer

    @property
    def weights(self):
        return self.FC_1.weights

    @weights.setter
    def weights(self, weights):
        self.FC_1.weights = weights

    @property
    def gradient_weights(self):

        return self.gradient_weights_FC1

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.FC_1.gradient_weights = value

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, state):
        self._memorize = state

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.weights)

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_2.initialize(weights_initializer, bias_initializer)
        self.FC_1.initialize(weights_initializer, bias_initializer)
