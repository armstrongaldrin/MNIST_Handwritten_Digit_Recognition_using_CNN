MNIST Handwritten Digit Recognition using CNN
This repository contains the implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras and achieves high accuracy on the test dataset.

Project Overview
The MNIST dataset contains 28x28 grayscale images of handwritten digits (0-9) with 60,000 training samples and 10,000 test samples. The goal of this project is to train a CNN to correctly classify the images of digits into their respective labels.


To run the project, you need the following dependencies:

Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
Jupyter Notebook
You can install the dependencies using pip:

pip install tensorflow keras numpy matplotlib
Project Structure

├── MNIST_Handwritten_Digit_Recognition_using_CNN.ipynb
├── README.md
└── dataset/
MNIST_Handwritten_Digit_Recognition_using_CNN.ipynb: The Jupyter notebook containing the entire code for data loading, model creation, training, evaluation, and visualization.
dataset/: Directory to store the MNIST dataset.
Model Architecture
The CNN architecture used in this project consists of:

Convolutional layers
MaxPooling layers
Dropout layers for regularization
Fully connected (Dense) layers
Output layer with softmax activation for classification
Training the Model
The model is trained using the categorical cross-entropy loss function and the Adam optimizer. It is evaluated using accuracy as the metric.

Parameters:
Epochs: 10
Batch size: 128
To start training, execute the Jupyter notebook.

Evaluation
The model is evaluated on the test dataset to calculate the accuracy and loss.

Results
The trained model achieves an accuracy of approximately 98% on the test dataset.

Usage
Clone this repository:
git clone https://github.com/armstrongaldrin/MNIST-Handwritten-Digit-Recognition-using-CNN.git
Install the required libraries:

pip install tensorflow keras numpy matplotlib
Run the notebook:

jupyter notebook MNIST_Handwritten_Digit_Recognition_using_CNN.ipynb
Train the model by following the steps in the notebook.
References
TensorFlow Documentation
Keras Documentation
