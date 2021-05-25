class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    # weight update the weight and implement the Stochastic Gradient Descent Algorithm
    # w_k+1 = w_k - n*gradient
    def calculate_update(self, weight_tensor, gradient_tensor):
        update_w = weight_tensor - (self.learning_rate * gradient_tensor)
        return update_w
