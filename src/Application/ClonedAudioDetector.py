from torch import nn


class CNNClassifier(nn.Module):
    """
    CNN Classifier with proper normalization and debugging
    (like fern's mana measurements but for neural nets)
    """

    def __init__(self, no_channels: int):
        super(CNNClassifier, self).__init__()
        self._channels = no_channels

        # Reduce dimensionality more gradually
        self.conv = nn.Sequential(
            # First block
            nn.Conv2d(self._channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            # Global pooling to reduce dimensions
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Much smaller fully connected layers
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
