import numpy as np
from evaluation import *
from activation_functions import *
'''
features    layer1         layer2
x1          node1
x2          node2
x3          node3           node1
...         ...
xn          noden   

weight_layer1.shape: (node last layer(features), node current layer)  
weight_layer2.shape: (node last layer1, node current layer(layer2))  
'''


class DeepLearning:
    def __init__(self, data, label, node_layer1, num_iteration, activation_function, node_layer2=1, learning_rate=0.01):
        self.data = data
        self.samples_cnt, self.features_cnt = self.data.shape
        self.y = label.reshape(self.samples_cnt, 1)
        self.weight_layer1 = np.random.rand(self.features_cnt, node_layer1) * np.sqrt(1. / self.features_cnt)
        self.weight_layer2 = np.random.rand(node_layer1, node_layer2) * np.sqrt(1. / node_layer1)
        # self.weight_layer3 = np.random.rand(node_layer2, 1)
        self.bias2 = np.random.rand(1, node_layer2)
        self.bias1 = np.random.rand(1, node_layer1)
        self.learning_rate = learning_rate
        self.activation = ActivationFunctions()
        self.num_iteration = num_iteration
        self.activation_function = activation_function

    def activation_fun(self, x):
        if self.activation_function == 1:
            return self.activation.sigmoid(x)
        else:
            return self.activation.relu(x)

    def derivative_activation_fun(self, x):
        if self.activation_function == 1:
            return self.activation.derivative_sigmoid(x)
        else:
            return self.activation.derivative_relu(x)

    def forward(self):
        # z1 = np.sum(w1 * x) + b1
        z1 = np.dot(self.data, self.weight_layer1) + self.bias1
        # a1 = activate_function(z1)
        a1 = self.activation_fun(z1)

        # z2 = np.sum(w2 * a1) + b2
        z2 = np.dot(a1, self.weight_layer2) + self.bias2
        # a2 = activate_function(z2)
        a2 = self.activation_fun(z2)
        return a1, a2

    def loss(self, y_pred):
        # J = -sum(y*logy' + (1-y)*log(1-y'))
        epsilon = 1e-10
        cost = -(self.y * np.log(y_pred + epsilon) + (1-self.y) * np.log(1-y_pred + epsilon))
        return np.mean(cost)

    def derivative_loss(self):
        _, y_pred = self.forward()
        error = y_pred - self.y
        return error/y_pred*(1-y_pred)

    def clip_gradients(self, gradient):
        return np.clip(gradient, -1, 1)  # Clipping gradients to be between -1 and 1

    def backward(self):
        for i in range(self.num_iteration):
            a1, y_pred = self.forward()
            eval = Evaluation(y=self.y, y_pred=y_pred)
            # dw2 = dj/da2 * da2/dz2 * dz2/dw2
            # dw2 = dj/da2 * da2/dz2 * dz2/db2
            delta2 = self.derivative_loss() * self.derivative_activation_fun(y_pred)

            dw2 = np.dot(a1.T, delta2)
            dw2 = self.clip_gradients(dw2)
            db2 = np.sum(delta2, keepdims=True, axis=0)
            db2 = self.clip_gradients(db2)

            # dw1 = dj/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
            delta1 = np.dot(delta2, self.weight_layer2.T) * self.derivative_activation_fun(a1)
            dw1 = np.dot(self.data.T, delta1)
            db1 = np.sum(delta1, keepdims=True, axis=0)
            dw1 = self.clip_gradients(dw1)
            db1 = self.clip_gradients(db1)

            # wi = wi - learningrate * dwi
            self.weight_layer2 -= self.learning_rate * dw2
            self.bias2 -= self.learning_rate * db2

            self.weight_layer1 -= self.learning_rate * dw1
            self.bias1 -= self.learning_rate * db1
            if i % 200 == 0:
                print(f'{i}: loss: {self.loss(y_pred)}, \
                accuracy: {eval.accuracy()}, \
                precision: {eval.precision()}, recall: {eval.recall()}, \
                f1_score: {eval.f1_score()}')

# import pandas as pd
# path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
# df = pd.read_csv(path)
# x = df[df.columns[:-1]]
# y = df[df.columns[-1]]
# Odl = DeepLearning(x.to_numpy(), y.to_numpy(), node_layer1=5 ,num_iteration=1000, learning_rate=0.001, node_layer2=1, activation_function=1)
# print(Odl.backward())