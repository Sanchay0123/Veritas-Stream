import torch
import torch.nn as nn
import torch.nn.functional as F

class Meso4(nn.Module):
    """
    MesoNet: a Compact Facial Video Forgery Detection Network.
    Paper: https://arxiv.org/abs/1809.00888
    """
    def __init__(self, num_classes=1):
        super(Meso4, self).__init__()
        
        # Layer 1: Conv2d -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2, 2)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)

        # Layer 3
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # Layer 4
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        # Fully Connected Layers (The Decision Maker)
        # We assume input image is resized to 256x256
        self.fc1 = nn.Linear(16 * 16 * 16, 16) 
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)