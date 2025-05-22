# Handwritten-Digit-Classifier

## Overview
This project utilizes a Convolutional Neural Network (CNN) to recognize handwritten digits from images. Specifically, it implements the LeNet architecture, which is known for its effectiveness in early image classification tasks.

#### LeNet Architecture Overview:
The network begins by applying convolutional filters over the input image. These filters help extract important features such as edges, borders, and loops. This step allows the model to understand the visual structure of the digits.

After each convolution, a ReLU (Rectified Linear Unit) activation is applied. ReLU introduces non-linearity into the model, enabling it to learn complex patterns beyond simple linear relationships.

Next, max pooling is performed to reduce the spatial dimensions of the feature maps. This helps in reducing computational complexity and also makes the detected features more robust to slight translations in the image.

The two-dimensional feature maps are then flattened into a one-dimensional vector. This flattened data serves as the input to the fully connected (dense) layers of the neural network.


| Files    | Description                                                                                                                   |
|----------|-------------------------------------------------------------------------------------------------------------------------------|
| train.py | This file is used to train the CNN off of the MNIST dataset.<br/>The model's weights are stored in model.pt                   | 
| test.py  | This file is used to test the models accuracy. It can be used to test from the MNIST dataset or from uploaded images as well. |
  | model.py| This file holds a class of the actual Lenet model.|


## Results
Within these dense layers, the network uses learned weights and hidden units to make a prediction about the digit in the image. During training, backpropagation is used to adjust the weights based on the error between the predicted and actual labels. This iterative process is how the model learns to improve its accuracy over time.
This model learned from the MNIST dataset of digits, which is provided by the PyTorch library
On the test dataset, the model performed at accuracy of 95.8%
