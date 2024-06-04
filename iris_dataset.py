import pandas as pd 
from normalization import *
from sklearn.datasets import load_iris


# load dataset
# path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
iris = load_iris()
x = iris.data
y = iris.target

# Create a boolean mask where only classes 0 and 1 are True
# mask = y != 2  
# x = x[mask]
# y = y[mask]


# normalize dataset
norm = NormalizeData()
x_norm = norm.z_score(x)

from deeplearning import *
Od1l = DeepLearning(x_norm, y)
layers = Od1l.layers(6)
Od1l.train(1000)