from deeplearning import *
import csv
import pandas as pd
import numpy as np
from normalization import *


class main:
    def __init__(self):
        self.normalize = None

    def show_menu(self):
        choose = 'y'
        while choose.lower() == 'y':
            print('1. Sigmoid function')
            print('2. Relu function')
            # print('3. Softmax function')
            # print('4. Tanh function')
            try:
                activation_function = int(input('Choose one from the list of activation functions: '))
                # node_layer1 = int(input('Import number of nodes in first layer (hidden layer): '))
                node_layer1 = 3
                # learning_rate = float(input('enter learning rate: '))
                learning_rate = 0.001
                # path = input('enter tha path of the file: ')
                path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
                # iteration = int(input('enter the number of iterations: '))
                print()
                iteration = 1000
                print('1. Max Abs Scaling')
                print('2. Robust Scaler')
                print('3. Z-Score Normalization')
                print('4. Min-Max Normalization')
                norm_func = int(input('select normalization function:'))
            except ValueError as e:
                print("Invalid input, please enter a number. Error: ", e)
                continue
            df = self.load_data(path)
            self.data_prepration(df, node_layer1, learning_rate, iteration, activation_function, norm_func)
            choose = input('Do you want to continue (y/n)?')
        else:
            print('bye!')

    def normalize_data(self, norm_func):
        if norm_func == 1:
            return self.normalize.max_abs()
        elif norm_func == 2:
            return self.normalize.robust_scale()
        elif norm_func == 3:
            return self.normalize.z_score()
        else:
            return self.normalize.min_max_func()

    def load_data(self, file_name):
        return pd.read_csv(file_name)

    def data_prepration(self, df, node_layer1, learning_rate, iteration, activation_function, norm_func):
        x = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        self.normalize = NormalizeData(x)
        x = self.normalize_data(norm_func)
        # x.apply()
        # print(x)

        Odl = DeepLearning(x.to_numpy(), y.to_numpy(), node_layer1=node_layer1, \
                           num_iteration=iteration, learning_rate=learning_rate, \
                           node_layer2=1, activation_function=activation_function)
        return Odl.backward()


if __name__ == '__main__':
    obj = main()
    obj.show_menu()
    # path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
