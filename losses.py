import numpy as np
from sklearn.metrics import log_loss

def cross_entropy_loss(y_pred, y_target, deriv = False):
    '''y_target is a one hot encoded vector, y_pred is the output from softmax layer'''
    
    if deriv:
        # assume that the last layer was softmax activated, then this equation is true
        return y_pred - y_target
    else:
       return log_loss(y_target, y_pred)


def main():
    y_target = np.array([[1,0,0]])
    y_pred = np.array([[0.2,0.7,0.1]])
    print(cross_entropy_loss(y_pred,y_target))

if __name__ == '__main__':
    main()