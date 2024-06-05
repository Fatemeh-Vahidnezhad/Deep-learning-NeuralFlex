# Deep Neural Network Framework

This project provides a lightweight and flexible Python framework for building and training deep neural networks.
It includes various activation functions, loss functions, and normalization techniques to cater to different neural network architectures and needs.

The deep learning framework is designed to handle a wide range of supervised learning tasks, including classification problems. Here are the specific types of classification tasks our model supports:

*Binary Classification: The model can distinguish between two classes. This is useful for yes/no type decisions, such as determining whether an email is spam or not spam.
      
*Multiclass Classification: The model is capable of classifying inputs into multiple categories. This is applicable to scenarios such as image recognition where each input may be categorized into one of       several predefined classes.


# Components
Loss Functions:
This module includes various loss functions essential for training deep learning models, supporting both calculation and derivation of loss metrics.

      Functions included:
            cross_Entropy_Loss()
            categorical_Cross_Entropy_Loss()
            mean_squared_error()
            derivative_cross_Entropy_Loss()
            derivative_categorical_Cross_Entropy_Loss()
**Evaluation**
The evaluation module provides functions to assess the accuracy and performance of the models.

**Metrics provided**

    Included functions:
            recall()
            accuracy()
            precision()
            F1_score()
            precision_multiclass_macro()

**Deep Learning**
Central to the framework, this component orchestrates the creation, training, and backpropagation for neural networks with a structure of one hidden layer and one output layer.

Key functionalities:
Handling of data and target labels.
Definition of the network architecture (number of nodes in the hidden layer).
Implementation of training procedures and backward propagation.


**Activation Functions**
Defines various activation functions used within neural networks, each with its derivative for use in backpropagation.

    Included functions:
          sigmoid()
          softmax()
          ReLU()
          Tanh()
          Derivatives for each activation function.
    
**Data Handling**
   Sample Dataset: A module to handle sample datasets for testing and demonstration purposes.
    Normalization: Methods to normalize data, which is critical for the efficient training of deep learning models.
    
    Included functions:
          min_max_func()
          z_score()
          max_abs()
          robust_scale()

## Installation

Clone this repository to your local machine

Navigate to the cloned repository

Install the following libraries:

 pip install numpy
 
 pip install sklearn
 
 pip install pandas
 

## Usage

To start using the framework, you can run one of the sample datasets as follows:

```bash
python iris_dataset.py
```

## Contact
fatemeh.vahidnezhad@gmail.com
