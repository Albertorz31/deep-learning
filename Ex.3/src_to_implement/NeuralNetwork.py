from copy import deepcopy


class NeuralNetwork:
    def __init__(self, optimizer=None, weights_initializer=None, bias_initializer=None):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer:
                regularization_loss = self.optimizer.regularizer.norm(layer.weights)
                loss += regularization_loss

        loss += self.loss_layer.forward(input_tensor, self.label_tensor)
        return loss

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for layer in self.layers:
            layer.testing_phase = False

        for _ in range(iterations):
            self.forward()
            self.backward(self.label_tensor)

    def test(self, input_tensor):
        for layer in self.layers:
            layer.testing_phase = True

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
