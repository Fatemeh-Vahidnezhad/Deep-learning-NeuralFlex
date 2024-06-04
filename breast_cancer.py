import numpy as np
from sklearn.datasets import load_breast_cancer
from normalization import *


# Load dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Display data structure
# print("Features shape:", X)

print("Target shape:", len(np.unique(y)))
# print("Classes:", data.target_names)
norm = NormalizeData()
x_norm = norm.z_score(x)

from deeplearning import *
Od1l = DeepLearning(x_norm, y)
layers = Od1l.layers(6)
Od1l.train(1000)
