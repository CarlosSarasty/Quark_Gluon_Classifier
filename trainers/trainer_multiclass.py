import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from jetnet.utils import to_image
from utils.utils import save_checkpoint
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from metrics.classification import ClassificationMetrics
from torch.optim.lr_scheduler import StepLR
from loss import get_loss

def log_metrics_to_tensorboard(metrics_dict, writer, epoch):
    for metric_name, metric_values in metrics_dict.items():
        writer.add_scalars(metric_name, metric_values, epoch)
        print(f'Adding {metric_name} {metric_values}')

def train_model(model, train_loader, val_loader, jet_type, **kwargs):
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
    images = kwargs.pop('images')

    if device == 'cuda':
        print(f'Device name: {torch.cuda.get_device_name(device)}')
    else: 
        print(f'Device {device}')

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    model.to(device)


    # Create an instance of MetricsTracker
    metrics  = ClassificationMetrics(5,jet_type)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize the scheduler 
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    im_size = 32
    maxR = 0.4

    for epoch in range(num_epochs):
        metrics.new_epoch()
        model.train()
        train_loss = 0.0
        total_train = 0
   
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            if images:
                inputs = to_image(inputs,  im_size = im_size, maxR = maxR)
                inputs  = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
               
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            train_loss += loss.item() * inputs.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_metric = metrics.train_batch_metrics(outputs, labels)
            total_train += labels.size(0)
        
        # Calculate and log training loss
        train_loss = train_loss / total_train

        # Print the training loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")


        scheduler.step() 
        val_loss = 0.0
        total_val = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                if images:
                    inputs = to_image(inputs,  im_size = im_size, maxR = maxR)
                    inputs  = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item() * inputs.size(0)
                valid_batch_metric = metrics.valid_batch_metrics(outputs, labels)
                total_val += labels.size(0)

        
        # Calculate and log validation loss
        _val = val_loss/total_val 
        dic_loss = {'Loss/Train': train_loss, 'Loss/Valid': _val }
        writer.add_scalars('Loss', dic_loss, epoch)
        
        # Print the validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {_val:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()[0]}")
       
        # Print the current learning rate
        if _val  < best_val_loss:
            best_val_loss = _val
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
    

    
        final_result = metrics.epoch_metrics()
        log_metrics_to_tensorboard(final_result, writer, epoch=epoch)
        writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch)

    writer.close()
               


