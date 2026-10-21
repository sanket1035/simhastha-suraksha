"""Baseline 1: MLP that predicts the people count directly from a small image."""
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def get_activation(name):
    try:
        return ACTIVATIONS[name]()
    except KeyError:
        raise ValueError("activation must be one of %s" % sorted(ACTIVATIONS))


class CountMLP(nn.Module):
    def __init__(self, input_size=128, hidden=(512, 128), activation="relu", dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = 3 * input_size * input_size
        for h in hidden:
            layers += [nn.Linear(in_features, h), get_activation(activation), nn.Dropout(dropout)]
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
