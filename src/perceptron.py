"""Single-layer perceptron from scratch (NumPy).

Used for two things: the classic AND/OR gate demo, and a tiny 'is this zone
unsafe?' classifier on (count, area). The rule density >= threshold is a
straight line in (count, area) space, so a perceptron can learn it.
"""
import numpy as np


class Perceptron:
    def __init__(self, n_features, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        return (np.asarray(X) @ self.w + self.b >= 0).astype(int)

    def fit(self, X, y, epochs=100):
        """Classic perceptron rule. Returns the number of mistakes per epoch."""
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        history = []
        for _ in range(epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                pred = int(xi @ self.w + self.b >= 0)
                error = yi - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    mistakes += 1
            history.append(mistakes)
            if mistakes == 0:
                break
        return history

    def accuracy(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
