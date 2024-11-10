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
            nn.Dropout(0.1),
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Global pooling to reduce dimensions
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Much smaller fully connected layers
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Add activation debugging
        debug = {}

        x = self.conv(x)
        debug["post_conv_mean"] = x.abs().mean().item()
        debug["post_conv_std"] = x.std().item()

        x = x.view(x.size(0), -1)
        debug["flattened_shape"] = list(x.shape)

        x = self.fc(x)
        debug["output_mean"] = x.abs().mean().item()
        debug["output_std"] = x.std().item()

        return x, debug if self.training else x
