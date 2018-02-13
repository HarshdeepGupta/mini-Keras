import numpy as np

def sigmoid(x, deriv = False):
    '''Calculates the sigmoidal activation function for a numpy array like input, or its derivative'''
    if deriv:
        return (x)*(1 -(x))
    else:
        return 1.0/(1.0 + np.exp(-x)) 


def relu(x, deriv = False, leak = 0.1):
    '''Calculates the relu activaiton function for a numpy array like input, or its derivative'''
    if deriv:
        return (x>0)*1
    else:
        return (x>0)*x


def softmax(x, deriv = False):
    if deriv:
        # TODO
        return Null
    else:
        maxx = np.max(x,axis = 1,keepdims=True)
        expos = np.exp(x - maxx)
        return expos/np.sum(expos, axis = 1, keepdims=True)

def linear(x, deriv = False):
    if deriv:
        return np.ones(x.shape)
    else:
        return x

def activation(x, type = 'sigmoid', deriv = False):
    if type == 'sigmoid':
        return sigmoid(x, deriv)
    elif type == 'relu':
        return relu(x,deriv)
    elif type == 'softmax':
        return softmax(x, deriv)
    elif type == 'linear' :
        return linear(x,deriv)
    else :
        raise ValueError("Not a valid activation function")

def main():
    # x = np.array([[1,2,3,-1]])
    # W = np.array([[1,2,3],[2,4,5],[2,3,4],[3,4,5]])
    # print(x.shape)
    # print(sigmoid(np.dot(x,W)))
    # print(softmax(np.dot(x,W)))

    x = np.array([[1,2,-3,8,9]])
    print(relu(x, deriv = True))



if __name__ == '__main__':
    main()