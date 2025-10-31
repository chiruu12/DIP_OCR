import torch.nn as nn


class CNNModel_Small(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel_Small, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer1(x); out = self.layer2(out); out = self.flatten(out)
        out = self.fc1(out); out = self.dropout(out); out = self.fc2(out)
        return out