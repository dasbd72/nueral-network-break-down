import unittest

import numpy as np
import torch

from .linear import Linear


class TestLinear(unittest.TestCase):
    def test_call(self):
        w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([7, 8], dtype=np.float64)
        x = np.array([[9, 10, 11], [12, 13, 14]], dtype=np.float64)

        layer = Linear(3, 2)
        layer.weight = w.copy()
        layer.bias = b.copy()
        y = layer(x)
        np.testing.assert_array_almost_equal(y, layer.forward(x), decimal=5)

    def test_forward(self):
        w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([7, 8], dtype=np.float64)
        x = np.array([[9, 10, 11], [12, 13, 14]], dtype=np.float64)

        layer = Linear(3, 2)
        layer.weight = w.copy()
        layer.bias = b.copy()
        y = layer.forward(x)

        pt_layer = torch.nn.Linear(3, 2)
        pt_layer.weight.data = torch.tensor(w)
        pt_layer.bias.data = torch.tensor(b)
        pt_x = torch.tensor(x)
        pt_y: torch.Tensor = pt_layer(pt_x)
        np.testing.assert_array_almost_equal(
            pt_y.detach().numpy(), y, decimal=5
        )

    def test_backward(self):
        w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([7, 8], dtype=np.float64)
        x = np.array([[9, 10, 11], [12, 13, 14]], dtype=np.float64)
        grad_output = np.array([[15, 16], [17, 18]], dtype=np.float64)

        layer = Linear(3, 2)
        layer.weight = w.copy()
        layer.bias = b.copy()
        y = layer.forward(x)
        grad = layer.backward(grad_output)

        pt_layer = torch.nn.Linear(3, 2)
        pt_layer.weight.data = torch.tensor(w)
        pt_layer.bias.data = torch.tensor(b)
        pt_x = torch.tensor(x, requires_grad=True)
        pt_y: torch.Tensor = pt_layer(pt_x)
        np.testing.assert_array_almost_equal(
            pt_y.detach().numpy(), y, decimal=5
        )
        pt_y.backward(torch.tensor(grad_output))
        np.testing.assert_array_almost_equal(pt_x.grad, grad, decimal=5)

    def test_update(self):
        w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([7, 8], dtype=np.float64)
        g_w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        g_b = np.array([7, 8], dtype=np.float64)
        lr = 0.5

        layer = Linear(3, 2)
        layer.weight = w.copy()
        layer.bias = b.copy()
        layer.weight_param.grad = g_w.copy()
        layer.bias_param.grad = g_b.copy()
        layer.update(lr)
        np.testing.assert_array_almost_equal(layer.weight, w - lr * g_w)
        np.testing.assert_array_almost_equal(layer.bias, b - lr * g_b)

        pt_layer = torch.nn.Linear(3, 2)
        pt_layer.weight.data = torch.tensor(w)
        pt_layer.bias.data = torch.tensor(b)
        pt_layer.weight.grad = torch.tensor(g_w)
        pt_layer.bias.grad = torch.tensor(g_b)
        pt_optim = torch.optim.SGD(pt_layer.parameters(), lr=lr)
        pt_optim.step()
        np.testing.assert_array_almost_equal(
            pt_layer.weight.detach().numpy(), layer.weight, decimal=5
        )
        np.testing.assert_array_almost_equal(
            pt_layer.bias.detach().numpy(), layer.bias, decimal=5
        )

    def test_zero_grad(self):
        w = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = np.array([7, 8], dtype=np.float64)

        layer = Linear(3, 2)
        layer.weight_param.grad = w.copy()
        layer.bias_param.grad = b.copy()
        layer.zero_grad()
        np.testing.assert_array_almost_equal(
            layer.weight_param.grad, np.zeros_like(w)
        )
        np.testing.assert_array_almost_equal(
            layer.bias_param.grad, np.zeros_like(b)
        )


if __name__ == "__main__":
    unittest.main()
