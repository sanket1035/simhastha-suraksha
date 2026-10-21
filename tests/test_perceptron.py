import unittest

import numpy as np

from src.perceptron import Perceptron


class PerceptronTests(unittest.TestCase):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

    def test_learns_and_gate(self):
        p = Perceptron(2)
        p.fit(self.X, [0, 0, 0, 1], epochs=50)
        self.assertEqual(p.accuracy(self.X, [0, 0, 0, 1]), 1.0)

    def test_learns_or_gate(self):
        p = Perceptron(2)
        p.fit(self.X, [0, 1, 1, 1], epochs=50)
        self.assertEqual(p.accuracy(self.X, [0, 1, 1, 1]), 1.0)

    def test_cannot_learn_xor(self):
        p = Perceptron(2)
        p.fit(self.X, [0, 1, 1, 0], epochs=100)
        self.assertLess(p.accuracy(self.X, [0, 1, 1, 0]), 1.0)

    def test_learns_unsafe_zone_rule(self):
        rng = np.random.default_rng(3)
        area = rng.uniform(50, 400, 600)
        count = rng.uniform(0, 1800, 600)
        density = count / area
        keep = np.abs(density - 4.0) > 0.3  # keep a margin so the data is cleanly separable
        X = np.column_stack([count[keep] / 100.0, area[keep] / 100.0])
        y = (density[keep] >= 4.0).astype(int)
        p = Perceptron(2, lr=0.1)
        p.fit(X, y, epochs=500)
        self.assertGreaterEqual(p.accuracy(X, y), 0.97)


if __name__ == "__main__":
    unittest.main()
