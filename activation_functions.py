import numpy as np


class ActivationFunctions:
    def sigmoid(self, x):                           # binary classification
        return 1/(1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        epsilon = 1e-10
        return np.where(x > 0, 1.0, 0.0 + epsilon)

    def softmax(self, x):                           # multiclass classification
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        z = np.exp(shifted_x)
        # normalize the matrix
        return z/z.sum(axis=1, keepdims=True)  
    
    def derivative_softmax(self, s):
        #shape of s: (m, n) --> reshape (m*n,1)
        s = s.reshape(-1,1)    
        #  np.diagflat(s)  # si * (1-si)
        #  np.dot(s, s.T)   # -si*sj --> i!=j
        return np.diagflat(s) - np.dot(s, s.T)

    def tanh(self, x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


# obj = ActivationFunctions()
# # Example usage
# z = np.array([[1.0, 2.0, 3.0],
#               [4.0, 7.0, 6.0]])
# s = obj.softmax(z)

# # print("Softmax Probabilities:")
# # print(s)

# # Compute the derivative (Jacobian matrix of the softmax)
# jacobian_matrix = obj.derivative_softmax(s)

# print("Jacobian Matrix of the softmax:")
# print(jacobian_matrix)