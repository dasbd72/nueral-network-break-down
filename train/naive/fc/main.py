import numpy as np
import torchvision

from nn.naive.modules.activation import ReLU
from nn.naive.modules.linear import Linear


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

    def update(self, lr: float) -> None:
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()


def train(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    iterations: int,
    batch_size: int,
    learning_rate: float,
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
        model.update(learning_rate)
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

    iterations = 1000
    batch_size = 32
    learning_rate = 0.01

    model = Model(28 * 28, 200, 10)
    train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        iterations,
        batch_size,
        learning_rate,
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
