from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, no_channels: int):
        super(CNNClassifier, self).__init__()
        self._channels = no_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self._channels, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        #TODO: figure out if we need softmax or not
        self.fc = nn.Sequential(nn.Linear(39744, 512), nn.ReLU(),
                                nn.Linear(512, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
