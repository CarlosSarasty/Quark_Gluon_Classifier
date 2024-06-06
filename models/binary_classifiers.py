import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 1)          # Output layer

        self.bn1 = nn.BatchNorm1d(32)

    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc2(x)


        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return torch.sigmoid(x)



class BinaryClassifier_v2(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier_v2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)         # Second hidden layer
        self.fc3 = nn.Linear(32, 1)          # Output layer

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)


        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return torch.sigmoid(x)


class BinaryClassifier_v3(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier_v3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, 32)         # tirth hidden layer
        self.fc4 = nn.Linear(32, 1)          # Output layer

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return torch.sigmoid(x)


class BinaryClassifierResidual(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifierResidual, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        # First residual block
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity  # Skip connection
        out = F.relu(out)

        # Second residual block
        identity = out
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.bn4(self.fc4(out))
        out += identity  # Skip connection
        out = F.relu(out)

        # Output layer
        out = self.fc5(out)
        return torch.sigmoid(out)
