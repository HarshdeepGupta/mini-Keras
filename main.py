import numpy as np
import time
from Sequential import Sequential
from Layer import Layer
 


def main():
    model = Sequential()
    model.add(Layer(size = 2)) # Input layer
    model.add(Layer(size = 6,activation = 'relu')) # Hidden layer with 6 neurons
    model.add(Layer(size = 6,activation = 'relu')) # Hidden layer with 6 neurons
    model.add(Layer(size = 2,activation = 'softmax')) # Output layer
    model.compile(learning_rate = 0.1)
    # learn the XOR mapping
    X =  np.array([[1,0],[0,1],[0,0],[1,1]])
    Y =  np.array([[1,0],[1,0],[0,1],[0,1]])

    print(model.predict(X))
    model.fit(X,Y,iterations =10000)
    print(model.predict(X))
      
if __name__ == "__main__":
    main()
