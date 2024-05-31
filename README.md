# Deep Neural Network Framework

This project provides a lightweight and flexible Python framework for building and training deep neural networks.
It includes various activation functions, loss functions, and normalization techniques to cater to different neural network architectures and needs.

## Features

- **Activation Functions**: Includes Sigmoid, ReLU, and options to extend with additional functions.
- **Normalization Techniques**: MaxAbs, RobustScaler, Z-Score, and Min-Max normalization.
- **Loss Functions**: Supports basic binary cross-entropy; extensible for other types such as MSE or Huber loss.
- **Evaluation Metrics**: Provides accuracy, precision, recall, and F1-score for model evaluation.
- **Optimization Algorithms**: Basic gradient descent with options to expand to Adam or RMSprop.

## Installation

Clone this repository to your local machine

Navigate to the cloned repository

Install the following libraries:
 pip install numpy
 pip install pandas

## Usage

To start using the framework, you can run the `main.py` script as follows:

```bash
python main.py
```

You can modify `main.py` to customize the network configuration, training parameters, and data paths according to your needs.

## Examples

Below is a simple example of how to use the framework:

```python
from deeplearning.activation_functions import ActivationFunctions
from deeplearning.deep_learning import DeepLearning

# Example data (X as inputs and y as labels)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Initialize the model
model = DeepLearning(data=X, label=y, node_layer1=5, num_iteration=1000, learning_rate=0.001, activation_function=1)

# Train the model
model.backward()

# Predict
predictions = model.forward()
```

## Contact
fatemeh.vahidnezhad@gmail.com
