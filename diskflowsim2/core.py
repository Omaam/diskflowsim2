"""Core module.
"""
import numpy as np

import utils


def compute_propagation_ratio(x, w):
    return softmax(x[..., None] * w)


def compute_propagation_ratio_with_radius(x, w, r):
    return softmax(x[..., None] * r[..., None, None] * w)


def propagate_conv(x):
    """Propagates with convolution.

    Propagates the cells in the given array by convolving with
    a kernel and rolling it.
    """
    num_cells_r = x.shape[0]
    y = utils.im2col_array(x[None, None, :], kernel_size=(1, 2),
                           stride=1, pad=(0, 1))
    y = np.dot(y, [1., 1.])
    y = y.reshape((num_cells_r, -1))
    y = y[:, 1:]
    y = np.roll(y, shift=1, axis=0)
    return y
