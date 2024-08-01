import torch
import yaml 
import numpy as np
import pandas as pd
from models.get_model import get_model



#Making a pandas dataframe 
def get_dataframes(jet_data, particle_data, data_args): # 30 particles by default
    n_particles = 100
    #Jet features
    jet_df = pd.DataFrame(jet_data, columns=data_args['jet_features'])
    
    #Particle features 
    reshaped_data = particle_data.reshape(-1, 3)
    means = np.mean(reshaped_data.reshape(-1, data_args['num_particles'], 3), axis=1)
    particle_df = pd.DataFrame(np.c_[jet_df['type'].values, means], columns=['type'] + data_args['particle_features'])

    ##### mixing all features
    all_feats_df = pd.DataFrame(np.c_[jet_data, means], columns=  data_args['jet_features']+data_args['particle_features'])

    return all_feats_df


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
   state = {
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state': optimizer.state_dict(),
   }
   torch.save(state, checkpoint_path)
   print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(config):

   #with open('config/config.yaml', 'r') as file:
   # config = yaml.safe_load(file)

   state = torch.load(checkpoint_path, weights_only=True)
   model = get_model(**config['model'])
   model.load_state_dict(state['model_state_dict'])
   #optimizer.load_state_dict(state['optimizer_state'])
   epoch = state['epoch']
   print(f"Checkpoint loaded from {checkpoint_path} at epoch {epoch}")
   return model #, optimizer, epoch # just the model for now
