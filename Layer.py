class Layer:
    '''
    A layer object for Sequential model which stores the size and activation function of the instantiated layer
    '''
    def __init__(self, size = 10, activation = 'linear'):
        self.size = size
        self.activation = activation