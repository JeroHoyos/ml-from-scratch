"""Activation functions for neural networks."""

import numpy as np


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def linear(x):
    """Linear activation function."""
    return x
