
import pickle
import matplotlib.pyplot as plt

def load_training_history(file_path):
    """
    Load the training history from a pickle file.#

    Parameters:
    - file_path: Path to the pickle file containing the training history.#

    Returns:
    - history: A dictionary containing training and validation losses and accuracies.
    """
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    return history


def plot_metrics(history, num_epochs, save_path=None):
    """
    Plot the training and validation loss and accuracy vs. epochs.

    Parameters:
    - history: A dictionary containing training and validation losses and accuracies.
    - num_epochs: Total number of epochs.
    - save_path: Path to save the plot as an image file. If None, the plot will be displayed.
    """
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accuracies = history['train_accuracies']
    val_accuracies = history['val_accuracies']
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plotting the training and validation loss
    plt.subplot(1, 2, 1)
    #plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

