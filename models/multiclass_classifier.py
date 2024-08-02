import torch 
import torch.nn as nn
import torch.nn.functional as F


class MulticlassClassifier(nn.Module):
    def __init__(self, model_type, input_dim, hidden_layers, output_dim, dropout_rate=0.5):
        super(MulticlassClassifier, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax to the output layer along the class dimension
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

