import numpy as np

def cross_entropy_loss(y_pred, y_target, deriv = False):
    '''y_target is a one hot encoded vector, y_pred is the output from softmax layer'''
    if deriv:
        return y_pred - y_target
    else:
        return -np.dot(y_target.T, np.log(y_pred))