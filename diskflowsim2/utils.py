"""Utilities.
"""
import numpy as np


def arrange_diskshape(x):
    y = np.fliplr(x)
    y = np.triu(y)
    y = np.fliplr(y)
    return y
