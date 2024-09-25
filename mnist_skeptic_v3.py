import torch
import torch.nn as nn
import torch.nn.functional as F

class skeptic_v3(nn.Module):
    def __init__(self):
        super(skeptic_v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 2 * 2, 256)  # Adjusted for 16x16 input
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        print(f"After first conv, batchnorm, and pool: {x.shape}")
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        print(f"After second conv, batchnorm, and pool: {x.shape}")
        x = self.pool(F.relu(self.conv3(x)))
        print(f"After third conv and pool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"After flattening: {x.shape}")
        x = F.relu(self.fc1(x))
        print(f"After first FC layer: {x.shape}")
        x = self.dropout(x)
        x = self.fc2(x)
        print(f"Final output shape: {x.shape}")
        return x