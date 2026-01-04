"""K-Nearest Neighbors classifier."""

import numpy as np


class KNN:
    """K-Nearest Neighbors classifier."""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Fit the KNN model (stores training data)."""
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """Make predictions."""
        # Implementation
        pass
