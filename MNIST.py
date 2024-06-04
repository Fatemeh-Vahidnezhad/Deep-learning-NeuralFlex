import tensorflow as tf
from normalization import *
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data: 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)  # -1 will calculate the necessary number of features (784 in this case)
x_test = x_test.reshape(x_test.shape[0], -1)

from deeplearning import *
Od1l = DeepLearning(x_train, y_train)
layers = Od1l.layers(6)
Od1l.train(20)