import torch
import torch.nn as nn
import torch.nn.functional as F

class CSNN(nn.Module):
    def __init__(self, num_channels=64, num_classes=2):
        super(CSNN, self).__init__()

        # Input: (batch, num_channels, n_times)
        # We'll reshape to (batch, 1, num_channels, n_times)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), stride=1, padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), stride=1, padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Placeholder for FC input size (initialized later)
        self.fc1 = None
        self.fc2 = None
        self.dropout = nn.Dropout(0.3)
        self.num_classes = num_classes

    def _initialize_fc(self, x):
        """Dynamically initialize the fully connected layers after first forward pass."""
        flatten_dim = x.view(x.size(0), -1).shape[1]
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc2 = nn.Linear(128, self.num_classes)
        # Move layers to same device as input
        self.fc1.to(x.device)
        self.fc2.to(x.device)

    def forward(self, x):
        # Input: (batch, n_channels, n_times)
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_times)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Initialize FC layers dynamically on first forward pass
        if self.fc1 is None:
            self._initialize_fc(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
