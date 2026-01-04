"""Data standardization techniques."""

import numpy as np


class StandardScaler:
    """Standardization (z-score normalization)."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """Fit the scaler."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        """Transform data."""
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)
