import argparse

import numpy as np
import torchvision

from nn.naive.modules.activation import ReLU
from nn.naive.modules.linear import Linear
from nn.naive.modules.optimizer import SGD, Adam, Optimizer
from nn.naive.modules.parameter import Parameter


class Args:
    iterations = 500
    batch_size = 32
    learning_rate = 0.01
    optimizer = "SGD"
    OPTIMIZERS = ["SGD", "Adam"]


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument(
    "--optimizer", type=str, default="SGD", choices=Args.OPTIMIZERS
)
args: Args = parser.parse_args()


class Model:
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int
    ):
        self.layers = [
            Linear(in_features, hidden_features),
            ReLU(),
            Linear(hidden_features, out_features),
        ]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def parameters(self) -> list[Parameter]:
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


def train(
    model: Model,
    optimizer: Optimizer,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    iterations: int,
    batch_size: int,
):
    for i in range(iterations):
        X_batch: np.ndarray
        y_batch: np.ndarray
        y_pred: np.ndarray

        indices = np.random.choice(len(X), batch_size)
        X_batch = X[indices].astype(np.float32)
        y_batch = y[indices].astype(np.float32)

        model.zero_grad()

        y_pred = model(X_batch)
        loss = np.mean((y_pred - y_batch) ** 2)
        grad_output = 2 * (y_pred - y_batch) / batch_size
        model.backward(grad_output)
        optimizer.step()
        print(f"Loss at iteration {i}: {loss}")

        if i % 10 == 0:
            y_val_pred = model(X_val)
            val_loss = np.mean((y_val_pred - y_val) ** 2)
            print(f"Validation loss at iteration {i}: {val_loss}")
            print()


def main():
    np.random.seed(0)

    # Already shuffled
    train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    X = train_data.data.numpy().reshape(-1, 28 * 28) / 255
    y = train_data.targets.numpy()

    y = np.eye(10)[y]
    X_train = X[:50000]
    y_train = y[:50000]
    X_val = X[50000:]
    y_val = y[50000:]

    iterations = args.iterations
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    model = Model(28 * 28, 200, 10)
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    train(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        iterations,
        batch_size,
    )

    print("Training set")
    y_pred = model(X_train)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_train, axis=1)
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy: {accuracy}")

    print("Validation set")
    y_pred = model(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
