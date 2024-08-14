import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x  # Save the input for the residual connection

        # Forward pass through the first layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Forward pass through the second layer
        out = self.fc2(out)
        out = self.bn2(out)

        # Add the identity (input) to the output of the second layer
        out += identity
        out = self.relu(out)
        out = self.out(out)
        return out

class autoencoder(nn.Module):
    def __init__(self, model_type, input_dim, output_dim, layers, dropout_rate=0.5):
        super(autoencoder, self).__init__()
        #self.layers = nn.ModuleList([ResidualBlock(k) for k in layers])
        self.input_layer = nn.Linear(input_dim, layers[0])
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.res1 = ResidualBlock(layers[0], layers[1])
        self.res2 = ResidualBlock(layers[1], layers[2])
        self.res3 = ResidualBlock(layers[2], layers[3])
        self.res4 = ResidualBlock(layers[3], layers[4])

        self.out = nn.Linear(layers[0], output_dim)
        self.bn_out = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        #x = self.dropout(x)
        x = self.res2(x)
        #x = self.dropout(x)
        x = self.res3(x)
        #x = self.dropout(x)
        x = self.res4(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.bn_out(x)
        return x
