"""Classification metrics: accuracy, precision, recall, f1, confusion_matrix."""

import numpy as np


def accuracy(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='binary'):
    """Calculate precision score."""
    # Implementation for precision
    pass


def recall(y_true, y_pred, average='binary'):
    """Calculate recall score."""
    # Implementation for recall
    pass


def f1_score(y_true, y_pred, average='binary'):
    """Calculate F1 score."""
    # Implementation for F1 score
    pass


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix."""
    # Implementation for confusion matrix
    pass
