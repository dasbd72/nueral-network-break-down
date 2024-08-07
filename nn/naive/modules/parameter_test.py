import unittest

import numpy as np

from .parameter import Parameter


class TestParameter(unittest.TestCase):
    def test_init(self):
        data = np.array([1, 2, 3])
        name = "param"
        param = Parameter(data, name)

        self.assertTrue(np.array_equal(param.data, data))
        self.assertTrue(np.array_equal(param.grad, np.zeros_like(data)))
        self.assertEqual(param.name, name)
        self.assertIs(param.data, data)

    def test_zero_grad(self):
        data = np.array([1, 2, 3])
        param = Parameter(data)

        param.grad = np.array([4, 5, 6])
        param.zero_grad()
        np.testing.assert_array_equal(param.grad, np.zeros_like(data))


if __name__ == "__main__":
    unittest.main()
