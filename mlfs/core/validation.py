"""Validation utilities for shapes, types, and data checks."""

import numpy as np


def check_X_y(X, y):
    """Validate X and y arrays."""
    if not isinstance(X, (list, np.ndarray)):
        raise TypeError("X must be a list or numpy array")
    if not isinstance(y, (list, np.ndarray)):
        raise TypeError("y must be a list or numpy array")
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    return X, y


def check_array(X):
    """Validate X array."""
    if not isinstance(X, (list, np.ndarray)):
        raise TypeError("X must be a list or numpy array")
    return np.asarray(X)


def check_is_fitted(estimator, attributes):
    """Check if estimator is fitted."""
    if not hasattr(estimator, attributes):
        raise ValueError("Estimator has not been fitted yet")
