import numpy as np
import time
from Sequential import Sequential
from Layer import Layer
 


def main():
    model = Sequential()
    model.add(Layer(size = 2))
    model.add(Layer(size = 6,activation = 'sigmoid'))
    model.add(Layer(size = 2,activation = 'softmax'))
    model.compile()
    # learn the XOR mapping
    X =  np.array([[1,0],[0,1],[0,0],[1,1]])
    Y =  np.array([[1,0],[1,0],[0,1],[0,1]])

    print(model.predict(X))
    model.fit(X,Y,iterations =10000)
    print(model.predict(X))
      
if __name__ == "__main__":
    main()
