import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, model_type, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: input channels = 1, output channels = 16, kernel size = 3x3
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        # Second convolutional layer: input channels = 16, output channels = 32, kernel size = 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Fully connected layer 1: 32*6*6 (after pooling) to 128
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        # Fully connected layer 2: 128 to num_classes
        self.fc2 = nn.Linear(128, num_classes)
        # Pooling layer: 2x2 window
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution -> ReLU -> Pooling
        x = x.view(-1, 32 * 6 * 6)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected layer -> ReLU
        x = self.fc2(x)  # Fully connected layer -> output logits
        return x
