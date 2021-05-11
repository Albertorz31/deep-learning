import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.input_tensor = None
        self.label_tensor = None
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.trainable = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        # Iterative over each layer and use
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        # The last layer = loss layer
        loss = self.loss_layer.forward(input_tensor, label_tensor)
        # Append loss into loss list
        self.loss.append(loss)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            # iterative over each layer
            input_tensor = layer.forward(input_tensor)
        return input_tensor
