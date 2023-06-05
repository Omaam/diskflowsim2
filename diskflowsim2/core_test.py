"""
"""
import unittest

import numpy as np

import core


class CoreTest(unittest.TestCase):

    def test_compute_propagation_ratio_0d(self):
        x = np.ones((1, 1))
        w = np.array([0.3, -0.4, 0.2])
        out = core.compute_propagation_ratio(x, w)
        actual = np.ones(x.shape[0])
        self.assertTrue(np.allclose(np.sum(out, axis=-1), actual))

    def test_compute_propagation_ratio_1d(self):
        x = np.ones((5, 5))
        w = np.array([0.3, -0.4, 0.2])

        out = core.compute_propagation_ratio(x, w)
        actual = np.ones(x.shape[0])
        self.assertTrue(np.allclose(np.sum(out, axis=-1), actual))

    def test_propagate_conv(self):
        x = np.arange(25).reshape(5, 5)
        print("******************")
        print(x)
        y = core.propagate_conv(x)
        print(y)
        print("******************")


if __name__ == "__main__":
    unittest.main()
