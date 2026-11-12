"""
Vectorized Logistic Regression with Gradient Descent from scratch using NumPy.
Author: Person A
Role: Discriminative Model
"""

import numpy as np


class ScratchLogisticRegression:
    """
    Logistic Regression with Gradient Descent and L2 regularization.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    n_iters : int, default=1000
        Maximum number of iterations.
    l2_penalty : float, default=0.0
        L2 regularization parameter (lambda).
    """

    def __init__(self, learning_rate=0.01, n_iters=1000, l2_penalty=0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    def fit(self, X, y):
        """
        Fit Logistic Regression model via gradient descent.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        eps = 1e-15  # Avoid log(0)

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Binary cross entropy loss with L2 regularization
            loss = -(1.0 / n_samples) * np.sum(
                y * np.log(np.clip(y_pred, eps, 1.0 - eps)) +
                (1.0 - y) * np.log(np.clip(1.0 - y_pred, eps, 1.0 - eps))
            ) + (self.l2_penalty / (2.0 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            # Gradient calculation
            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2_penalty / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        p1 = self._sigmoid(linear_model)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
