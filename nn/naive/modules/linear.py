import numpy as np

from .parameter import Parameter


class Linear:
    def __init__(self, in_features: int, out_features: int):
        """Creates a Linear layer with in_features input features and out_features output features.
        Initializes the weights using He initialization and sets the biases to zero. (Important !!!)

        :param int in_features: number of input features
        :param int out_features: number of output features
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features) * np.sqrt(
            2.0 / in_features
        )  # He initialization
        self.bias = np.zeros(out_features)
        self.weight_param = Parameter(self.weight, name="linear_weight")
        self.bias_param = Parameter(self.bias, name="linear_bias")
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
        return np.dot(x, self.weight.T) + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Performs backward pass through the layer.

        :param np.ndarray grad_output: gradient tensor of shape (batch_size, out_features)
        :return: gradient tensor of shape (batch_size, in_features)
        :rtype: np.ndarray
        """
        self.weight_param.grad = np.dot(grad_output.T, self.input)
        self.bias_param.grad = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weight)

    def zero_grad(self):
        """Resets the gradient tensors to zero."""
        self.weight_param.zero_grad()
        self.bias_param.zero_grad()

    def parameters(self):
        """Returns the layer parameters.

        :return: list of parameters
        :rtype: list[Parameter]
        """
        return [
            self.weight_param,
            self.bias_param,
        ]
