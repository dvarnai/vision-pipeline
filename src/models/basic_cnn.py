from math import ceil

import torch
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, in_height: int, in_width: int):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # [B, 8, in_height // 2, in_width // 2]

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(16 * 4 * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def __repr__(self) -> str:
        return f'BasicCNN(num_classes={self.num_classes}, in_channels={self.in_channels})'

    def architecture(self):
        return str(self.features), str(self.classifier)

    def __call__(self, x):
        return self.forward(x)