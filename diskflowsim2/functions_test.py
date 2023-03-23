"""
"""
import unittest

import numpy as np

import functions as func


class FunctionsTest(unittest.TestCase):
    def test_softmax_1d(self):
        x = np.array([[0.3, 0.4, 0.1]])
        out = func.softmax(x)
        actual = np.array([1.0, 1.0])
        self.assertTrue(np.allclose(np.sum(out, axis=1), actual))

    def test_softmax_2d(self):
        x = np.array([[0.3, -0.4, 1.1],
                      [0.1, -0.1, 0.1]])
        out = func.softmax(x)
        actual = np.array([1.0, 1.0])
        self.assertTrue(np.allclose(np.sum(out, axis=1), actual))


if __name__ == "__main__":
    unittest.main()
