class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    # weight update the weight and implement the Stochastic Gradient Descent Algorithm
    # W_n+1 = W - n*(gradient(L(W_n))
    def calculate_update(self, weight_tensor, gradient_tensor):
        update_w = weight_tensor - (self.learning_rate * gradient_tensor)
        return update_w
