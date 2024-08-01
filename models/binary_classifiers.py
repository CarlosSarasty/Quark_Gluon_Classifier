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
        x = self.fc2(x)


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


class QuarkGluonClassifierWithEmbeddings(nn.Module):
    def __init__(self, num_particles, embedding_dim, jet_input_dim, hidden_dim, output_dim):
        super(QuarkGluonClassifierWithEmbeddings, self).__init__()
        self.embedding = nn.Linear(3, embedding_dim)  # Linear embedding for particle features
        self.fc1 = nn.Linear(jet_input_dim + num_particles * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, jet_features, particle_features):
        # Embed particle features
        embedded_particles = self.embedding(particle_features)
        embedded_particles = embedded_particles.view(embedded_particles.size(0), -1)  # Flatten the embeddings
        # Concatenate jet features with particle embeddings
        x = torch.cat((jet_features, embedded_particles), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
