import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from utils.utils import load_checkpoint
from models.get_model import get_model
    

def evaluate_and_plot(val_loader, checkpoint_path, save_dir, train_name):
    """
    Evaluate the model and plot the confusion matrix and ROC curve.

    Parameters:
    - model: Pretrained PyTorch model.
    - val_loader_path: DataLoader for the validation dataset.
    - save_dir: Directory to save the plots.
    """


    with open('config/config.yaml', 'r') as file:
      config = yaml.safe_load(file)

    name = config['inference']['train_name']
 
    checkpoint = torch.load(config['inference']['checkpoint_path'])
    model = get_model(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])   
    model.eval()


    # Lists to store true labels and predictions
    true_labels = []
    predictions = []
    predicted_probs = []

    # No gradient is needed for evaluation
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.float())
            probs = outputs.squeeze()
            predicted_labels = (probs >= 0.5).float()  # Convert probabilities to binary labels
            true_labels.extend(labels.tolist())
            predictions.extend(predicted_labels.tolist())
            predicted_probs.extend(probs.tolist())

    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions, normalize='true')


    # Ensure save_dir exists 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Save the confusion matrix
    conf_matrix_path = f'{save_dir}/confusion_matrix_{name}.pt'
    torch.save(conf_matrix, conf_matrix_path)
    print(f"Confusion matrix saved to {conf_matrix_path}")

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{save_dir}/confusion_matrix_{name}.png')
    plt.show()

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_dir}/roc_curve_{name}.png')
    plt.show()
