"""
"""
import unittest

import numpy as np

import core


class CoreTest(unittest.TestCase):
    def test_softmax_1d(self):
        x = np.array([[0.3, 0.4, 0.1]])
        out = core.softmax(x)
        actual = np.array([1.0, 1.0])
        self.assertTrue(np.allclose(np.sum(out, axis=1), actual))

    def test_softmax_2d(self):
        x = np.array([[0.3, -0.4, 1.1],
                      [0.1, -0.1, 0.1]])
        out = core.softmax(x)
        actual = np.array([1.0, 1.0])
        self.assertTrue(np.allclose(np.sum(out, axis=1), actual))

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


if __name__ == "__main__":
    unittest.main()
