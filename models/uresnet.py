import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class UResNet(nn.Module):
    def __init__(self, model_type, input_dim, num_classes):
        super(UResNet, self).__init__()
        
        self.model_type = model_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        hidden_dim = 64  # Base hidden dimension

        # Encoder: Use input_dim and hidden_dim to define the layers
        self.enc1 = BasicBlock(input_dim, hidden_dim)
        self.enc2 = BasicBlock(hidden_dim, hidden_dim * 2)
        self.enc3 = BasicBlock(hidden_dim * 2, hidden_dim * 4)
        
        # Bottleneck
        self.bottleneck = BasicBlock(hidden_dim * 4, hidden_dim * 8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=2, stride=2)
        self.dec3 = BasicBlock(hidden_dim * 8, hidden_dim * 4)
        
        self.upconv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2)
        self.dec2 = BasicBlock(hidden_dim * 4, hidden_dim * 2)
        
        self.upconv1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        self.dec1 = BasicBlock(hidden_dim * 2, hidden_dim)
        
        # Final classification layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        
        # Bottleneck
        x = self.bottleneck(F.max_pool2d(x3, 2))
        
        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat((x, x3), dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.dec1(x)
        
        # Global average pooling to reduce to [N, hidden_dim, 1, 1]
        x = self.global_pool(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Final fully connected layer for classification
        x = self.fc(x)
        
        return x
