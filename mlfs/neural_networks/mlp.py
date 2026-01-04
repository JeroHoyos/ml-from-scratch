"""Multi-Layer Perceptron (MLP)."""

from .base import NeuralNetwork


class MLP(NeuralNetwork):
    """Multi-Layer Perceptron neural network."""
    
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, max_iter=200):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
    
    def fit(self, X, y):
        """Train the MLP."""
        pass
    
    def predict(self, X):
        """Make predictions."""
        pass
