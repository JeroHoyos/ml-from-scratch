"""Data normalization techniques."""

import numpy as np


class MinMaxScaler:
    """Min-Max normalization (0-1 scaling)."""
    
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit(self, X):
        """Fit the scaler."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        """Transform data."""
        return (X - self.min_) / (self.max_ - self.min_)
    
    def fit_transform(self, X):
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)
