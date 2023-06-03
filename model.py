from sklearn.neural_network import MLPClassifier
import numpy as np
from activations import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    tanh, tanh_derivative
)


class MLPClassifier(MLPClassifier):
    ACTIVATION_FUNCTIONS = {
        "logistic": sigmoid,
        "tanh": tanh,
        "relu": relu
    }
    ACTIVATION_DERIVATIVE_FUNCTIONS = {
        "logistic": sigmoid_derivative,
        "tanh": tanh_derivative,
        "relu": relu_derivative
    }

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        alpha=0.0001,
        batch_size=200,
        learning_rate=0.001,
        max_iter=200,
        epsilon=1e-8,
    ):
        self._act_func = self.ACTIVATION_FUNCTIONS[activation]
        self._act_der_func = self.ACTIVATION_DERIVATIVE_FUNCTIONS[activation]
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.n_layers = len(hidden_layer_sizes) + 2
        self.layer_sizes = [None] + list(hidden_layer_sizes) + [None]

    def _activate(self, X):
        activations = [X if isinstance(X, np.ndarray) else X.toarray()]
        for layer in range(self.n_layers - 1):
            activations.append(
                self._act_func(
                    activations[layer] @ self.weights[layer] + self.biases[layer]
                )
            )
        return activations

    def _weighted_sum(self, layer):
        return self.activations[layer] @ self.weights[layer] + self.biases[layer]

    def fit(self, X, y):
        self.layer_sizes[0] = X.shape[1]
        self.layer_sizes[-1] = np.unique(y).shape[0]
        self.weights = [
            np.full((self.layer_sizes[layer], self.layer_sizes[layer + 1]), 0.0)
            for layer in range(self.n_layers - 1)
        ]
        self.biases = [
            np.zeros(self.layer_sizes[layer + 1]) for layer in range(self.n_layers - 1)
        ]

        for iteration in range(self.max_iter):
            batch_size = min(self.batch_size, X.shape[0])
            batch_indices = np.random.choice(X.shape[0], batch_size, replace=False)
            X_sample, y_sample = X[batch_indices], y[batch_indices]

            self.activations = self._activate(X_sample)
            y_prob = self.activations[-1]
            y_true = np.eye(self.layer_sizes[-1])[y_sample]
            loss = np.mean(np.sum((y_prob - y_true) ** 2, axis=1))

            grad_b = [None] * (self.n_layers - 1)
            grad_w = [None] * (self.n_layers - 1)
            l2 = self.alpha / 2 * np.sum(np.square(self.weights[-1]))
            loss_der_by_act = 2 * (self.activations[-1] - y_true) + l2
            for layer in range(self.n_layers - 2, -1, -1):
                loss_der_by_bias = loss_der_by_act * self._act_der_func(self._weighted_sum(layer))
                grad_b[layer] = np.sum(loss_der_by_bias, axis=0)
                grad_w[layer] = np.dot(self.activations[layer].T, loss_der_by_bias)

                if layer == 0: break
                loss_der_by_act = np.dot(
                    loss_der_by_act, self.weights[layer].T
                ) * self._act_der_func(self._weighted_sum(layer - 1))

            for layer in range(self.n_layers - 1):
                self.biases[layer] -= self.learning_rate * grad_b[layer]
                self.weights[layer] -= self.learning_rate * grad_w[layer]

            if loss < self.epsilon:
                break

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        return self._activate(X)[-1]
