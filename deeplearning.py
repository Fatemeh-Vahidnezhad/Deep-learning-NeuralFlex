import numpy as np
from evaluation import *
from activation_functions import *
from loss_func import *
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
    def __init__(self, data, labels, learning_rate=0.01):
        self.data = data#.to_numpy()
        self.num_classes = len(np.unique(labels))
        # label = label#.to_numpy()
        self.samples_cnt, self.features_cnt = self.data.shape
        self.y = labels.reshape(-1, 1) if self.num_classes == 2 else np.eye(self.num_classes)[labels]

        self.learning_rate = learning_rate
        self.activation = ActivationFunctions()
        # self.activation_function = activation_function
        self.loss = Loss() 

    def activation_fun(self, x):
        if self.num_classes == 2:
            return self.activation.sigmoid(x)
        # elif self.activation_function == 2:
        #     return self.activation.relu(x)
        else:
            return self.activation.softmax(x)

    def derivative_activation_fun(self, x):
        if self.num_classes == 2:
            return self.activation.derivative_sigmoid(x)
        else:
            return self.activation.derivative_softmax(x)

    def layers(self, node_layer1):
        self.node_layer1 = node_layer1
        self.weight_layer1 = np.random.rand(self.features_cnt, node_layer1) * np.sqrt(1. / self.features_cnt)
        node_layer2 = 1 if self.num_classes == 2 else self.num_classes
        self.weight_layer2 = np.random.rand(node_layer1, node_layer2) * np.sqrt(1. / node_layer1)
        

        # self.weight_layer3 = np.random.rand(node_layer2, 1)
        self.bias1 = np.random.rand(1, node_layer1)
        self.bias2 = np.random.rand(1, node_layer2)

        # layer1: z1 = np.sum(w1 * x) + b1
        z1 = np.dot(self.data, self.weight_layer1) + self.bias1
        # a1 = activate_function(z1)
        self.a1 = self.activation_fun(z1)

        # layer2: z2 = np.sum(w2 * a1) + b2
        z2 = np.dot(self.a1, self.weight_layer2) + self.bias2
        # a2 = activate_function(z2)
        self.y_pred = self.activation_fun(z2)
        # return a1, a2

    def loss_func(self):
        if self.num_classes == 2:
            return self.loss.cross_Entropy_Loss(self.y, self.y_pred)
        else:
            return self.loss.categorical_Cross_Entropy_Loss(self.y, self.y_pred)
        

    def derivative_loss(self):
        if self.num_classes == 2:
            return self.loss.derivative_cross_Entropy_Loss(self.y, self.y_pred)
        else:
            return self.loss.derivative_categorical_Cross_Entropy_Loss(self.y, self.y_pred)

    def clip_gradients(self, gradient):
        return np.clip(gradient, -1, 1)  # Clipping gradients to be between -1 and 1

    def backward(self):
        # dw2 = dj/da2 * da2/dz2 * dz2/dw2
        # db2 = dj/da2 * da2/dz2 * dz2/db2
        if self.num_classes == 2:
            delta2 = self.derivative_loss() * self.derivative_activation_fun(self.y_pred)
        else:
            delta2 = self.derivative_loss()
        dw2 = np.dot(self.a1.T, delta2)
        dw2 = self.clip_gradients(dw2)
        db2 = np.sum(delta2, keepdims=True, axis=0)
        db2 = self.clip_gradients(db2)

        # dw1 = dj/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
        if self.num_classes == 2:
            delta1 = np.dot(delta2, self.weight_layer2.T) * self.derivative_activation_fun(self.a1)
        else:
            delta1 = np.dot(delta2, self.weight_layer2.T) * self.a1 * (1 - self.a1)

        dw1 = np.dot(self.data.T, delta1)
        db1 = np.sum(delta1, keepdims=True, axis=0)
        dw1 = self.clip_gradients(dw1)
        db1 = self.clip_gradients(db1)

        # wi = wi - learningrate * dwi
        self.weight_layer2 -= self.learning_rate * dw2
        self.bias2 -= self.learning_rate * db2

        self.weight_layer1 -= self.learning_rate * dw1
        self.bias1 -= self.learning_rate * db1

    def train(self, num_iteration):
        for i in range(num_iteration):
            self.layers(self.node_layer1)
            self.backward()
            eval = Evaluation(y=self.y, y_pred=self.y_pred)
            if self.num_classes ==2:
                if i % 10 == 0:
                    print (f'{i}: loss: {self.loss_func()}, accuracy: {eval.accuracy()}, precision: {eval.precision()}, recall: {eval.recall()},f1_score: {eval.f1_score()}')
            else:
                if i % 10 == 0:
                    print (f'{i}: loss: {self.loss_func()}, accuracy: {eval.accuracy()}, precision, recall: {eval.precision_mulitclass_macro()}')
            



# import pandas as pd
# path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
# df = pd.read_csv(path)
# x = df[df.columns[:-1]]
# y = df[df.columns[-1]]
# Odl = DeepLearning(x.to_numpy(), y.to_numpy(), node_layer1=5 ,num_iteration=1000, learning_rate=0.001, node_layer2=1, activation_function=1)
# print(Odl.backward())