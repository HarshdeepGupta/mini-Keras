import numpy as np
import activations 
import losses
import Layer as L

class Sequential:

    def __init__(self):
        '''
        Initialize the model with instance variables
        '''
        # layer_sizes[i] stores the size of ith layer of the model
        # weight_matrices and biases at ith index store the weights going from ith layer to (i+1)th layer
        # non_lins[i] stores the activation function of the ith layer
        self.layer_sizes = []
        self.weight_matrices = []
        self.biases = []
        self.non_lins = []

    def add(self, layer):
        '''
        Add layers to the model
        '''
        self.layer_sizes.append(layer.size)
        self.non_lins.append(layer.activation)
        if len(self.layer_sizes) > 1:            
            size_of_prev_layer = self.layer_sizes[-2]
            size_of_curr_layer = self.layer_sizes[-1]
            self.weight_matrices.append(np.random.rand(size_of_prev_layer,size_of_curr_layer))
            self.biases.append(np.ones([1,size_of_curr_layer]))
            



    def predict(self, x):
        '''
        Calculates the output of the model for the input x
        '''
        for layer in range(0,len(self.layer_sizes)-1):
            # use activations from the previous layer, and nonlinearities of the current layer
            x = activations.activation(  np.dot(x, self.weight_matrices[layer] ) + self.biases[layer] , self.non_lins[layer+1]) 
        return x
    
    def compile(self, learning_rate = 0.01, regularization = 'ridge', alpha = 0.01):
        '''
        sets the hyper parameters of the model, alpha is the shrinkage constant
        '''
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.alpha = alpha
    
    def fit(self,x, targets, iterations = 100):
        '''
        fits the model on the given data by using Gradient Descent for specified iterations
        '''
        for _ in range(iterations):

            # Forward propagation : non_lin( matrix multiplication + biases)
            layer_activations = [x]
            for layer in range(0,len(self.layer_sizes)-1):
                curr_layer_activation = activations.activation(  np.dot(layer_activations[layer], self.weight_matrices[layer] ) +
                                        self.biases[layer] ,
                                        self.non_lins[layer+1]) 
                layer_activations.append(curr_layer_activation)
            # Backward propagation : calculate the errors and gradients
            deltas = [None] * (len(self.layer_sizes))
            # we assume that loss is always cross entropy and last layer is a softmax layer
            deltas[-1] = losses.cross_entropy_loss(layer_activations[-1],targets,deriv = True)
            # start the iteration from the second last layer 
            for layer in range(len(deltas)-2,0,-1):
                deltas[layer] = np.dot(deltas[layer+1], self.weight_matrices[layer].T) * activations.activation( layer_activations[layer],
                                                        type = self.non_lins[layer], deriv = True) 

            # Calculate the weight and bias updates
            weight_updates = []
            bias_updates = []
            for layer in range(1,len(layer_activations)):
                weight_updates.append(np.dot(layer_activations[layer-1].T, deltas[layer]))
                bias_updates.append(np.sum(deltas[layer], axis=0,keepdims=True))
            
            for layer in range(len(self.weight_matrices)):
                self.weight_matrices[layer] += -self.learning_rate*weight_updates[layer]
                self.biases[layer] += -self.learning_rate*bias_updates[layer]



    # def train(self, x, targets, iters, learning_rate = 0.001, alpha = 0.01):

def main():
    model = Sequential()
    model.add(L.Layer(size = 2))
    model.add(L.Layer(size = 6,activation = 'sigmoid'))
    model.add(L.Layer(size = 2,activation = 'softmax'))
    model.compile()

    X =  np.array([[1,0],[0,1],[0,0],[1,1]])
    Y =  np.array([[1,0],[1,0],[0,1],[0,1]])

    print(model.predict(X))
    model.fit(X,Y,iterations =10000)
    print(model.predict(X))


if __name__ == '__main__':
    main()