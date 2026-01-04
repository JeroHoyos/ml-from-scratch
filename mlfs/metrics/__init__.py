"""Metrics module for evaluation."""

from .classification import accuracy, precision, recall, f1_score, confusion_matrix
from .regression import mean_squared_error, mean_absolute_error, r2_score

__all__ = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
]
