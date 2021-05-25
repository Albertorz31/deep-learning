class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weight = None

    def forward(self, input_tensor):
        raise NotImplementedError

    def backward(self, error_tensor):
        raise NotImplementedError
