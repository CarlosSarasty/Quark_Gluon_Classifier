import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import save_checkpoint
import time

def train_model(model, train_loader, val_loader, **kwargs):
    '''
    Train the model with the given data loaders and configuration.

    Parameters:
    - model: The neural network model to train.
    - train_loader: DataLoader for the training dataset.
    - val_loader: DataLoader for the validation dataset.
    - **kwargs: Additional keyword arguments for training configuration.
      - num_epochs (int): Number of epochs to train. Default is 10.
      - learning_rate (float): Learning rate for the optimizer. Default is 0.001.
      - checkpoint_path (str): Path to save the best model checkpoint. Default is 'best_model.pth'.
      - log_dir (str): Directory for TensorBoard logs. Default is 'runs'.
      - device (str): Device to run the training on ('cpu' or 'cuda'). Default is 'cpu'.
    '''
      
    num_epochs = kwargs.pop('num_epochs')
    learning_rate = kwargs.pop('learning_rate')
    checkpoint_path = kwargs.pop('checkpoint_path')
    log_dir = kwargs.pop('log_dir')
    device = kwargs.pop('device')   
    if device == 'cuda':
        print(f'Device name: {torch.cuda.get_device_name(device)}')
    else: 
        print(f'Device {device}')
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    model.to(device)

    # Track losses and accuracies for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        train_start_time = time.time()  # Start timer for training

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs.squeeze() >= 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / total_train)
        train_accuracies.append(train_accuracy)
        writer.add_scalar('Loss/train', train_loss / total_train, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        train_time = time.time() - train_start_time  # Calculate training time

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_start_time = time.time()  # Start timer for validation

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs.squeeze() >= 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / total_val)
        val_accuracies.append(val_accuracy)
        writer.add_scalar('Loss/val', val_loss / total_val, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        val_time = time.time() - val_start_time  # Calculate validation time

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/total_train:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss/total_val:.4f}, Val Accuracy: {val_accuracy:.4f}, Train Time: {train_time:.2f} s, Val Time: {val_time:.2f} s')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
    
    writer.close()

    # Save losses and accuracies for further plotting if needed
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, 'training_stats.pth')

    # Save training history for plotting
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

    return history
