import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TabularEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the Decoder
class TabularDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(TabularDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        return x

# Define the Encoder-Decoder Model
class TabularAutoencoder(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim):
        super(TabularAutoencoder, self).__init__()
        self.encoder = TabularEncoder(input_dim, hidden_dim)
        self.decoder = TabularDecoder(hidden_dim, output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

