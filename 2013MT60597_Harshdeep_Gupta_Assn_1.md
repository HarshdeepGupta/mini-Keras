<center> 

ELL 888: Advanced Machine Learning
Assignment 1 

Submitted By:

Harshdeep Gupta - 2013MT60597

</center>

# Introduction

In this assignment, we implement a neural network in python from scratch.  We use the scientific computation library numpy to vectorize the calculations, and try to mimic the API provided by Keras for Sequential models.

We implement activation functions, backpropagation algorithm, regularization, loss functions, metrics for accuracy in code.  

# Code Walkthrough

The code is divided across different modules, with the main modules being:

* **Sequential.py**

  This is the main workhorse of the project, where the code for defining and training the model is written. It provides the `Sequential` class, which is used to define the model. Layers are added to model using the `add()` method. 

  ```python
  # code for creating and training a model
  model = Sequential()
  model.add(Layer(size = INPUT_SIZE))
  model.add(Layer(size = HIDDEN_LAYER_size, activation = 'sigmoid'))
  model.add(Layer(size = OUTPUT_SIZE, activation = 'softmax'))
  model.compile(learning_rate = 0.001, weight_penalty = 0.001, penalty = 'ridge')
  model.fit(X_train, Y_train, iterations = 100)
  ```

* **activations.py**

  This module implements the various activation functions, along with their gradients. The activation functions currently available are `sigmoid` ,`relu`  and `linear` .

* **losses.py**

  This module implements the loss functions. Since this assignment had classification task, currently it implements `cross_entropy_loss`

* **metrics.py**

  This module implements the various metrics, like accuracy , the user might want to monitor during the training and testing. It implements a zero-one loss function currently.

# Dataset

We use a subset of the **EMNIST** dataset for training and testing purposes. The six letters that we choose are *a,d,e,g,h,p* , selected as per the instructions mentioned in the assignment problem statement. The code for processing the data for training is present in `data/emnist_data_processing.ipynb` file. The following is a frequency analysis of the collected data. Each class has 4800 samples in training dataset, and 800 samples in test dataset. Images are 28*28 in dimensions, and labels are one hot encoded into six dimensional vectors, corresponding to six letters.

# Models

We use two models to plot our graphs

* one has a single hidden layer, we henceforth refer to it as the shallow network
* one has three hidden layers, we henceforth refer to it as the deep network

# Experiments

###  Train and Test error plots

|      |      |
| ---- | ---- |
|      |      |

