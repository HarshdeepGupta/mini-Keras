import numpy as np


def accuracy(y_pred, target):
    res = 0
    for i in range(len(target)):
        if np.argmax(y_pred[i]) == np.argmax(target[i]):
            res+=1
    return res/len(target)
   


def main():
    target = [[1,0],[0,1]]
    y_pred = [[0,1],[0,1]]
    print(accuracy(y_pred,target))

if __name__ == '__main__':
    main()