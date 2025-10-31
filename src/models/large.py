import torch.nn as nn

class CNNModel_Large(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel_Large, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv_block1(x); out = self.conv_block2(out); out = self.conv_block3(out)
        out = self.flatten(out); out = self.fc_block(out); out = self.classifier(out)
        return out
