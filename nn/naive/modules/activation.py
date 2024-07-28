import numpy as np


class ReLU:
    def __init__(self):
        """Creates a ReLU activation layer."""
        self.input = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Performs forward pass through the layer.

        :param np.ndarray x: input tensor of shape (batch_size, in_features)
        :return: output tensor of shape (batch_size, out_features)
        :rtype: np.ndarray
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs forward pass through the layer.

        :param np.ndarray x: input tensor of shape (batch_size, in_features)
        :return: output tensor of shape (batch_size, out_features)
        :rtype: np.ndarray
        """
        self.input = x.copy()
        return np.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Performs backward pass through the layer.

        :param np.ndarray grad_output: gradient tensor of shape (batch_size, out_features)
        :return: gradient tensor of shape (batch_size, in_features)
        :rtype: np.ndarray
        """
        return grad_output * (self.input > 0)
