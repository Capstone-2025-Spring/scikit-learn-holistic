import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=4):
        super(CNNClassifier, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 10, 83)
        x = self.conv_block(x)
        x = self.classifier(x)
        return x
