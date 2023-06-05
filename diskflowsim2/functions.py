"""Functions
"""
import numpy as np


def softmax(x, axis=-1):
    # For avoiding overflow.
    y = x - x.max(axis=axis, keepdims=True)
    y = np.exp(y)
    y /= y.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def distribute_zero_one(x):
    return (x - np.min(x)) / np.max(x)
