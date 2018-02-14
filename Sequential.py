import numpy as np
from activations import activation
import losses
from Layer import Layer
from metrics import accuracy

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
            self.weight_matrices.append(2*np.random.rand(size_of_prev_layer,size_of_curr_layer)-1)
            self.biases.append(0.01*np.ones([1,size_of_curr_layer]))
            



    def predict(self, x, return_activations = False):
        '''
        Calculates the output of the model for the input x
        If return_activations is set to true, then returns a python list of activations of all layers 
        '''
        # Forward propagation : non_lin( matrix multiplication + biases)
        layer_activations = [x]
        for layer in range(0,len(self.layer_sizes)-1):
            # use activations from the previous layer, and nonlinearities of the current layer
            curr_layer_activation = activation(  np.dot(layer_activations[layer], self.weight_matrices[layer] ) +
                                    self.biases[layer] ,
                                    self.non_lins[layer+1]) 
            layer_activations.append(curr_layer_activation)
        if return_activations:
            return layer_activations
        else :
            return layer_activations[-1] 
    
    def backward_pass(self, layer_activations, targets):
        '''
        Return the deltas for each layer of the network; deltas are as defined in theory of Michael Nielson's book 
        '''
        # Backward propagation : calculate the errors and gradients
        deltas = [None] * (len(self.layer_sizes))
        # we assume that loss is always cross entropy and last layer is a softmax layer
        deltas[-1] = losses.cross_entropy_loss(layer_activations[-1],targets,deriv = True)
        # start the iteration from the second last layer 
        for layer in range(len(deltas)-2,0,-1):
            deltas[layer] = np.dot(deltas[layer+1], self.weight_matrices[layer].T) * activation( layer_activations[layer],
                                                    type = self.non_lins[layer], deriv = True)
        return deltas

    def calculate_param_updates(self, layer_activations, deltas,batch_size):
        ''' 
        Calculate the weight and bias updates based on activations and deltas
        '''
        weight_updates = []
        bias_updates = []
        for layer in range(1,len(layer_activations)):
            # divide the updates by batch size 
            weight_updates.append(np.dot(layer_activations[layer-1].T, deltas[layer])/batch_size)
            bias_updates.append(np.sum(deltas[layer], axis=0,keepdims=True)/batch_size)
        return weight_updates, bias_updates


    def penalty(self, weight_matrix):
        '''
        return the regularization term for the corresponding weight matrix
        '''
        if self.regularization == 'ridge':
            return weight_matrix
        elif self.regularization == 'lasso':
            return np.sign(weight_matrix)
        else:
            raise ValueError('Invalid regularization scheme. Use "ridge" or "lasso"')
    
    def fit(self,x, targets, iterations = 100):
        '''
        fits the model on the given data by using Gradient Descent for specified iterations
        '''
        batch_size = x.shape[0]
        for _ in range(iterations):
            # Forward propagation : non_lin( matrix multiplication + biases)
            layer_activations = self.predict(x, return_activations = True)
            # Backward propagation : calculate the errors and gradients
            deltas = self.backward_pass(layer_activations,targets)
            # Calculate the weight and bias updates
            weight_updates, bias_updates = self.calculate_param_updates(layer_activations,deltas,batch_size)
            # Update the weight matrices and biases of the model
            for layer in range(len(self.weight_matrices)):
                self.weight_matrices[layer] += -self.learning_rate* ( weight_updates[layer] + self.weight_penalty * self.penalty(self.weight_matrices[layer]))
                self.biases[layer] += -self.learning_rate*bias_updates[layer]
    
    
    
    def compile(self, learning_rate = 0.01, regularization = 'ridge', weight_penalty = 0.001):
        '''
        sets the hyper parameters of the model, alpha is the shrinkage constant
        '''
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weight_penalty = weight_penalty
        # convert lists to arrays
        self.weight_matrices = np.array(self.weight_matrices)
        # self.biases = np.array(self.biases)
        self.layer_sizes = np.array(self.layer_sizes)
        self.non_lins = np.array(self.non_lins)
    
    def score(self,x,target, metric = 'accuracy'):
        y = self.predict(x)
        if metric == 'accuracy':            
            return accuracy(y,target)
        elif metric == 'loss':
            return losses.cross_entropy_loss(y,target)


def main():
    model = Sequential()
    model.add(Layer(size = 2))
    model.add(Layer(size = 6,activation = 'sigmoid'))
    model.add(Layer(size = 2,activation = 'softmax'))
    model.compile(learning_rate = 1)

    X =  np.array([[1,0],[0,1],[0,0],[1,1]])
    Y =  np.array([[1,0],[1,0],[0,1],[0,1]])

    print(model.predict(X))
    model.fit(X,Y,iterations =1000)
    print(model.predict(X))


if __name__ == '__main__':
    main()