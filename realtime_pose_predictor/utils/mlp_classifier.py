# utils/mlp_classifier.py

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=90, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
