import numpy as np
import time


def sigmoid(x, deriv = False):
    '''Calculates the sigmoidal activation function for a numpy array like input, or its derivative'''
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
        return Null
    else:
        expos = np.exp(x - np.max(x))
        return expos/np.sum(expos, axis = 1, keepdims=True)

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
        return y_pred - y_target
    else:
        return -np.dot(y_target.T, np.log(y_pred))

def forward_pass(x,w_1,w_2,b_1,b_2):
    # forward propagation : matrix multiplication + biases
    # print("x1 shape = {}".format(x.shape))
    # print("w1 shape = {}".format(w_1.shape))
    a_2 = np.dot(x,w_1) + b_1
    # print("a2 shape = {}".format(a_2.shape))
    z_2 = activation(a_2)
    # print("z2 shape = {}".format(z_2.shape))

    a_3 = np.dot(z_2, w_2) + b_2
    # print("a3 shape = {}".format(a_3.shape))
    z_3 = activation(a_3, 'softmax')
    y_pred = z_3
    return y_pred



def train(x,t,w_1,w_2,b_1,b_2,alpha = 0, learning_rate = 0.01):
    '''x is the input data,t is the target, v and w are layers, bv and bw are the biases'''
    for _ in range(1000000):
        # forward propagation : matrix multiplication + biases
        # print("x1 shape = {}".format(x.shape))
        # print("w1 shape = {}".format(w_1.shape))
        a_2 = np.dot(x,w_1) + b_1
        # print("a2 shape = {}".format(a_2.shape))
        z_2 = activation(a_2)
        # print("z2 shape = {}".format(z_2.shape))

        a_3 = np.dot(z_2, w_2) + b_2
        # print("a3 shape = {}".format(a_3.shape))
        z_3 = activation(a_3, 'softmax')
        y_pred = z_3
        # print("z3 shape = {}".format(z_3.shape))

        # calculate loss
        loss = cross_entropy_loss(y_pred,t)
        # print(loss)
        # calculate gradient of loss    
        delta_3 = cross_entropy_loss(y_pred,t,deriv = True)
        # print("delta3  = {}".format(delta_3))
        # Backward propagation : calculate the errors and gradients
        dw_2 = np.dot(a_2.T,delta_3)
        db_2 = np.sum(delta_3, axis=0, keepdims=True)
        delta_2 = np.dot(delta_3,w_2.T)* activation(z_2,deriv = True)
        # print("delta_2 shape = {}".format(delta_2.shape))
        dw_1 = np.dot(x.T, delta_2)
        db_1 = np.sum(delta_2, axis=0,keepdims=True)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dw_2 += alpha * w_2
        dw_1 += alpha * w_1

        # Do the gradient descent parameter update
        # print("w2 update = {}".format(-learning_rate*dw_2))
        w_2 += -learning_rate*dw_2
        w_1 += -learning_rate*dw_1
        b_1 += -learning_rate*db_1
        b_2 += -learning_rate*db_2

    return w_2,b_2,w_1,b_1




# network topology
INPUT_SIZE = 2
H1_SIZE = 6
OUTPUT_Size = 2
BATCH_SIZE = 4
# hyperparameter
learning_rate = 0.01
alpha = 0.1

np.random.seed(7)



def main():
    # x = np.array([[1,2,3,-1]])
    # W = np.array([[1,2,3],[2,4,5],[2,3,4],[3,4,5]])
    # print(x.shape)
    # print(sigmoid(np.dot(x,W)))
    # x = np.random.rand(BATCH_SIZE,INPUT_SIZE)
    x = np.array([[1,0],[0,1],[0,0],[1,1]])
    # target = np.random.rand(BATCH_SIZE,OUTPUT_Size)
    target = np.array([[1,0],[1,0],[0,1],[0,1]])
    w_1 = np.random.rand(INPUT_SIZE, H1_SIZE)
    w_2 = np.random.rand(H1_SIZE, OUTPUT_Size)
    b_1 = np.ones([1, H1_SIZE])
    b_2 = np.ones([1, OUTPUT_Size])
    
    print(forward_pass(x,w_1,w_2,b_1,b_2))
    w_2, b_2, w_1, b_1 = train(x,target,w_1,w_2,b_1,b_2)
    print(forward_pass(x,w_1,w_2,b_1,b_2))
if __name__ == "__main__":
    main()
