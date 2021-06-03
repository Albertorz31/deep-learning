import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    # weight update the weight and implement the Stochastic Gradient Descent Algorithm
    # w_k+1 = w_k - n*gradient
    def calculate_update(self, weight_tensor, gradient_tensor):
        update_w = weight_tensor - (self.learning_rate * gradient_tensor)
        return update_w

    class SgdWithMomentum:
        # Parameter update based on current and past gradients: momentum
        def __init__(self, learning_rate, momentum_rate):
            self.learning_rate = float(learning_rate)
            self.momentum_rate = float(momentum_rate)

        def calculate_update(self, weight_tensor, gradient_tensor):
            # v_k = momentum* v_k-1 - n*gradient
            try:
                self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
            except:
                self.v = 0
                self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
            return weight_tensor + self.v

    class Adam:
        # parameter update based on current and past gradients:
        def __init__(self, learning_rate, mu, rho):
            self.learning_rate = float(learning_rate)
            self.mu = float(mu)
            self.rho = float(rho)
            self.g = float(0)  # gradient of current iteration
            self.v = float(0)  # first order momentum term
            self.r = float(0)  # second order momentum
            self.count = float(1)

        def calculate_update(self, weight_tensor, gradient_tensor):
            try:
                # v_k = mu*v_k-1 + (1-mu)*gradient
                self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
                # r_k = rho*r_k-1 + (1-rho)* gradient x gradient (Multiply arguments element-wise)
                self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)
            except:
                self.v = 0
                self.r = 0
                self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
                self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

            # Bias correction
            v_hat_k = self.v / (1 - self.mu ** self.count)
            r_hat_k = self.r / (1 - self.rho ** self.count)
            self.count += 1
            w_k1 = weight_tensor - self.learning_rate * (v_hat_k + np.finfo(np.float).eps) / (
                        np.sqrt(r_hat_k) + np.finfo(np.float).eps)
            return w_k1
