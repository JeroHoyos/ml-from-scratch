"""Base classes for neural networks."""


class NeuralNetwork:
    """Base class for neural networks."""
    
    def __init__(self):
        self.layers = []
        self.parameters = {}
    
    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def fit(self, X, y):
        """Train the neural network."""
        pass
    
    def predict(self, X):
        """Make predictions."""
        pass
