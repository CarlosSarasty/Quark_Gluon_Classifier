from sklearn.model_selection import train_test_split    
from torch.utils.data import DataLoader, TensorDataset
import torch, numpy as np

def get_loaders(particle_data, jet_data):

    y = np.array(jet_data[:,0], dtype=int)
    # Determine the number of unique classes
    num_classes = np.max(y) + 1
    # Initialize the ground truth matrix with zeros
    ground_truth = np.zeros((y.size, int(num_classes)))
    # Set the appropriate indices to 1
    ground_truth[np.arange(y.size), y] = 1    
    X_train, X_test, y_train, y_test = train_test_split(particle_data, ground_truth, test_size=0.2, random_state=42)
    
    # Create datasets 
    train_dataset = TensorDataset(torch.tensor(X_train, dtype= torch.float32 ), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype= torch.float32 ), torch.tensor(y_test, dtype=torch.long))
    
    batch_size = 50
    train_loader = DataLoader(train_dataset , batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return  train_loader, val_loader
