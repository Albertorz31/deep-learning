class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weight = None
        self.testing_phase = False

    def forward(self, input_tensor):
        raise NotImplementedError

    def backward(self, error_tensor):
        raise NotImplementedError

