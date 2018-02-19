class Layer:
    '''
    A layer object for Sequential model which stores the size and activation function of the instantiated layer
    '''
    def __init__(self, size = 10, activation = 'linear', type = 'dense', dropout_keep_prob = 1):
        self.size = size
        self.activation = activation
        self.type = type
        self.dropout_keep_prob = dropout_keep_prob