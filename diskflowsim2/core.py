"""Core module.
"""
import numpy as np

import diskflowsim2 as dfs2
import utils


def compute_propagation_ratio(x, w):
    return dfs2.softmax(x[..., None] * w)


def compute_propagation_ratio_with_radius(x, w, r):
    return dfs2.softmax(x[..., None] * r[..., None, None] * w)


def propagate_conv(x):
    """Propagates with convolution.

    Propagates the cells in the given array by convolving with
    a kernel and rolling it.
    """
    num_cells_r = x.shape[0]

    y = utils.im2col_array(x[None, None, :], kernel_size=(1, 2),
                           stride=1, pad=(0, 1))

    kernel = np.array([0.5, 0.5])
    y = np.dot(y, kernel)
    y = y.reshape((num_cells_r, -1))

    y = y[:, 1:]

    y = np.roll(y, shift=1, axis=0)
    y[0] = 0

    return y
