"""Classification models."""

from .logistic import LogisticRegression
from .svm import SVM
from .knn import KNN
from .tree import DecisionTree

__all__ = ['LogisticRegression', 'SVM', 'KNN', 'DecisionTree']
