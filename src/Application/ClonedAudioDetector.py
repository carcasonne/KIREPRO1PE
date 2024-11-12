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
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            # Second block
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third block
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            # Fourth block
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Much smaller fully connected layers
        self.fc = nn.Sequential(nn.Linear(43008, 512), nn.ReLU(), nn.Linear(512, 2), nn.Softmax(dim=1))


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
