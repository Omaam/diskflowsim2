"""Functions
"""
import numpy as np


def softmax(x, axis=None):
    # For avoiding overflow.
    y = x - x.max(axis=axis, keepdims=True)
    y = np.exp(y)
    y /= y.sum(axis=axis, keepdims=True)
    return y
