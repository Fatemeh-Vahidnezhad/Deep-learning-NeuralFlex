import numpy as np 


class Loss:
    def cross_Entropy_Loss (self, y, y_pred):
        # J = -sum(y*logy' + (1-y)*log(1-y'))
        epsilon = 1e-10
        cost = -(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))
        return np.mean(cost)

    def derivative_cross_Entropy_Loss(self, y, y_pred):
        error = y_pred - y
        return error/y_pred*(1-y_pred)

    def mean_squared_error(self, y, y_pred):
        # MSE = sum(y-y_pred)**2/m   m: number of samples
        return np.mean((y - y_pred)**2)
    
    def derivative_mean_squared_error(self, y, y_pred, x):
        # dMSE/dw = (-2/m) * sum((y_pred - y)*xi)
        m = len(x)
        return -2 * np.dot(x.T, (y_pred - y))/m
    
    def categorical_Cross_Entropy_Loss(self, y, y_pred):
        return -np.sum(y * np.log(y_pred))
    
    def derivative_categorical_Cross_Entropy_Loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -y/y_pred
    

    



