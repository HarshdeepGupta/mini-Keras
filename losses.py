import numpy as np

def cross_entropy_loss(y_pred, y_target, deriv = False):
    '''y_target is a one hot encoded vector, y_pred is the output from softmax layer'''
    if deriv:
        # assume that the last layer was softmax activated, then this equation is true
        return y_pred - y_target
    else:
        return -np.sum(y_target* np.log(y_pred),axis = 1,keepdims=True)


def main():
    y_target = np.array([[1,0,0]])
    y_pred = np.array([[0.2,0.7,0.1]])
    print(cross_entropy_loss(y_pred,y_target))

if __name__ == '__main__':
    main()