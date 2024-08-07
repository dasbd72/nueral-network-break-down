import numpy as np


class Parameter(object):
    def __init__(self, data: np.ndarray, name: str = ""):
        """Creates a parameter with data and grad tensors that points to the same memory location.

        :param np.ndarray data: data tensor
        :param np.ndarray grad: gradient tensor
        """
        self.data = data
        self.grad = np.zeros_like(data)
        self.name = name

    def zero_grad(self) -> None:
        """Zeros the gradient tensor."""
        self.grad.fill(0)
