"""Categorical encoding techniques."""

import numpy as np


class OneHotEncoder:
    """One-hot encoding for categorical variables."""
    
    def __init__(self):
        self.categories_ = None
    
    def fit(self, X):
        """Fit the encoder."""
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        """Transform data."""
        # Implementation
        pass
    
    def fit_transform(self, X):
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    """Label encoding for categorical variables."""
    
    def __init__(self):
        self.classes_ = None
    
    def fit(self, y):
        """Fit the encoder."""
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y):
        """Transform data."""
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])
    
    def fit_transform(self, y):
        """Fit and transform."""
        self.fit(y)
        return self.transform(y)
