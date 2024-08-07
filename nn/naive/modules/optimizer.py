import numpy as np

from .parameter import Parameter


class Optimizer(object):
    def __init__(
        self,
        parameters: list[Parameter],
    ):
        """Initializes the optimizer.

        :param parameters: list of parameters to optimize
        """
        self.parameters = parameters

    def step(self):
        """Performs an update step for all parameters."""
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        parameters: list[Parameter],
        lr=0.01,
    ):
        """Initializes the optimizer.

        :param parameters: list of parameters to optimize
        :param lr: learning rate
        """
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        """Updates parameters using SGD."""
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Parameter],
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        """Initializes the optimizer.

        :param parameters: list of parameters to optimize
        :param lr: learning rate
        :param beta1: beta1 parameter for Adam
        :param beta2: beta2 parameter for Adam
        :param epsilon: epsilon parameter for Adam
        """
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Updates parameters using Adam."""
        self.t += 1
        for i, parameter in enumerate(self.parameters):
            self.m[i] = (
                self.beta1 * self.m[i] + (1 - self.beta1) * parameter.grad
            )
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (
                parameter.grad**2
            )

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            parameter.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
