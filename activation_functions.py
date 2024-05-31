import numpy as np


class ActivationFunctions:
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        epsilon = 1e-10
        return np.where(x > 0, 1.0, 0.0 + epsilon)

    def softmax(self, x):
        z = x - np.max(x, axis=1, keepdims=True)
        return np.exp(z)/np.sum(np.exp(z, axis=1, keepdims=True))

    def derivative_softmax(self):
        pass

    def tanh(self, x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


