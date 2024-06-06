import yaml

from jetnet.datasets import JetNet
from utils.utils import *
from data.get_data_loaders import get_data_loaders
from models.get_model import get_model
from trainers.trainer import train_model
from scripts.plot_metrics import plot_metrics
from scripts.evaluate_model import evaluate_and_plot

import yaml
import argparse
import JetNet
# import QuarkGluon  # Uncomment if you use QuarkGluon instead of JetNet
from your_model_file import get_dataframes, get_data_loaders, get_model, train_model, plot_metrics, evaluate_and_plot

def launch_project(config_file):
    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    data_args = config['data']['data_args']

    n_part = data_args['num_particles']
    sav = config['training']['checkpoint_path']
    print(f'Running with {n_part} particles per jet, and saving {sav}')

    # Get particles and jet data
    particle_data, jet_data = JetNet.getData(**data_args)
    # particle_data, jet_data = QuarkGluon.getData(**data_args)  # Uncomment if you use QuarkGluon

    # Make pandas dataframes for easy data manipulation
    all_feats_df = get_dataframes(jet_data, particle_data, data_args)

    # Get data loaders
    train_loader, val_loader = get_data_loaders(all_feats_df, data_args)
    model = get_model(**config['model'])

    # Train the model
    history = train_model(model, train_loader, val_loader, **config['training'])

    # Plot metrics
    num_epochs = len(history['train_losses'])
    plot_metrics(history, num_epochs)  # '/content/drive/MyDrive/Project/Loss_Accuracy.png')

    # Evaluate the model
    # model_path = config['inference']['model_path']
    evaluate_and_plot(val_loader, **config['inference'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch project with configuration file.')
    parser.add_argument('config_file', type=str, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    launch_project(args.config_file)
