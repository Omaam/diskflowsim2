"""
"""
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def compute_propagation_ratio(x, w):
    return softmax(x[..., None] * w)


def compute_propagation_ratio_with_radius(x, w, r):
    return softmax(x[..., None] * r[..., None, None] * w)
