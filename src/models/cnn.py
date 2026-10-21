"""Baseline 2: small CNN that predicts the people count directly."""
import torch.nn as nn

from .mlp import get_activation


class CountCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), activation="relu", dropout=0.0):
        super().__init__()
        layers, in_c = [], 3
        for c in channels:
            layers += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c), get_activation(activation), nn.MaxPool2d(2)]
            in_c = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(in_c * 16, 128), get_activation(activation),
                                  nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(1)
