import numpy as np
import time


def sigmoid(x, deriv = False):
    '''Calculates the sigmoidal activaiton function for a numpy array like input, or its derivative'''
    if deriv:
        return (x)*(1 -(x))
    else:
        return 1.0/(1.0 + np.exp(-x)) 


def relu(x, deriv = False, leak = 0.1):
    '''Calculates the relu activaiton function for a numpy array like input, or its derivative'''
    if deriv:
        if x >= 0:
            return 1
        else:
            return -leak
    else:
        if x >= 0:
            return x
        else :
            return leak*x


def softmax(x, deriv = False):
    if deriv:
        # TODO
        return NULL 
    else:
        expos = np.exp(x - np.max(x))
        return expos/np.sum(expos)

def activation(x, type = 'sigmoid', deriv = False):
    if type == 'sigmoid':
        return sigmoid(x, deriv)
    elif type == 'relu':
        return relu(x,deriv)
    elif type == 'softmax':
        return softmax(x, deriv)
    else :
        raise ValueError("Not a valid activation function")

def cross_entropy_loss(y_pred, y_target, deriv = False):
    '''y_target is a one hot encoded vector, y_pred is the output from softmax layer'''
    if deriv:
        # TODO
        return y_target - y_pred
    else:
        return -np.dot(y_target, np.log(y_pred))

# network topology
INPUT_SIZE = 784
L1_SIZE = 500
L2_SIZE = 400
OUTPUT_Size = 6

# hyperparameters
learning_rate = 0.01


np.random.seed(7)





def train(x,t,w_1,w_2,b_1,b_2):
    '''x is the input data,t is the target, v and w are layers, bv and bw are the biases'''
    # forward propagation : matrix multiplication + biases
    A_1 = np.dot(x,w_1) + b_1
    Z_1 = activation(A_1)

    A_2 = np.dot(Z_1, w_2) + b_2
    # TODO: implement softmax
    Z_2 = activation(A_2, 'softmax')
    Y = Z_2

    # TODO: calculate loss
    loss = cross_entropy_loss(Y,t)
    # TODO: calculate gradient of loss
    # delta_2 = 

    # Backward propagation : calculate the errors and gradients
    


    # calculate the loss






def main():
    x = np.array([[1,2,3,-1]])
    W = np.array([[1,2,3],[2,4,5],[2,3,4],[3,4,5]])
    print(x.shape)
    print(sigmoid(np.dot(x,W)))

if __name__ == "__main__":
    main()
