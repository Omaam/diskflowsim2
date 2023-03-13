"""Functions
"""
import numpy as np


def softmax(x):
    # For avoiding overflow.
    y = x - x.max(axis=-1, keepdims=True)
    y = np.exp(y)
    y /= y.sum(axis=-1, keepdims=True)
    return y
