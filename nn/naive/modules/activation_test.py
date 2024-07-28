import unittest

import numpy as np
import torch

from .activation import ReLU


class TestReLU(unittest.TestCase):
    def test_call(self):
        x = np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float64)

        relu = ReLU()
        np.testing.assert_array_equal(relu(x), relu.forward(x))

    def test_forward(self):
        x = np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float64)

        relu = ReLU()
        y = relu.forward(x)

        pt_relu = torch.nn.ReLU()
        pt_x = torch.tensor(x)
        pt_y: torch.Tensor = pt_relu(pt_x)
        np.testing.assert_array_equal(pt_y.detach().numpy(), y)

    def test_backward(self):
        x = np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float64)
        grad_output = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        relu = ReLU()
        y = relu.forward(x)
        grad = relu.backward(grad_output)

        pt_relu = torch.nn.ReLU()
        pt_x = torch.tensor(x, requires_grad=True)
        pt_y: torch.Tensor = pt_relu(pt_x)
        np.testing.assert_array_equal(pt_y.detach().numpy(), y)
        pt_y.backward(torch.tensor(grad_output))
        np.testing.assert_array_equal(pt_x.grad, grad)


if __name__ == "__main__":
    unittest.main()
