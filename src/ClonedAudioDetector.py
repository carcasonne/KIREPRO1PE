from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, no_channels: int):
        super(CNNClassifier, self).__init__()
        self._channels = no_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self._channels, 32, kernel_size=2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
