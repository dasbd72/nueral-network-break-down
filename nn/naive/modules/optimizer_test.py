import unittest

import numpy as np

from .optimizer import SGD, Adam
from .parameter import Parameter


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        """Set up test parameters."""
        # Initialize test parameters with specific values
        self.data = np.array([1.0, 2.0, 3.0])
        self.grad = np.array([0.1, 0.1, 0.1])
        self.parameter = Parameter(data=self.data.copy())
        self.parameter.grad = self.grad.copy()

    def test_sgd_step(self):
        """Test the SGD optimizer's step function."""
        # Initialize SGD optimizer
        optimizer = SGD(parameters=[self.parameter], lr=0.1)
        # Perform an optimization step
        optimizer.step()
        # Expected updated data
        expected_data = self.data - 0.1 * self.grad
        # Assert that the parameter data is updated correctly
        np.testing.assert_almost_equal(self.parameter.data, expected_data)

    def test_adam_step(self):
        """Test the Adam optimizer's step function."""
        # Initialize Adam optimizer
        optimizer = Adam(parameters=[self.parameter], lr=0.1)
        # Perform an optimization step
        optimizer.step()
        # Calculate expected data using Adam formula for one step
        m = 0.1 * self.grad
        v = 0.001 * (self.grad**2)
        m_hat = m / (1 - 0.9)
        v_hat = v / (1 - 0.999)
        expected_data = self.data - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Assert that the parameter data is updated correctly
        np.testing.assert_almost_equal(self.parameter.data, expected_data)


if __name__ == "__main__":
    unittest.main()
