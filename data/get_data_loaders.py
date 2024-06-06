import torch, numpy as np
from data.customdataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data_loaders(all_feats_df, data_args):
  '''
    This function prepares the data loaders for training and validation from the given dataframe of features.

    Parameters:
    - all_feats_df (DataFrame): A pandas DataFrame containing the combined jet and particle features.
    - data_args (dict): A dictionary containing data-related arguments including feature names.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    '''

  feats = data_args['jet_features'][1:]+data_args['particle_features']

  y = all_feats_df['type'].values 
  y_bin = np.where(y == 2, 1, 0) # using binary labels 0 and 1

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(all_feats_df[feats].values, y_bin, test_size=0.2, random_state=42)

  # Scale the features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)


  # Create the dataset and data loader
  batch_size = 100
  train_dataset = CustomDataset(X_train_scaled, y_train)
  train_loader = DataLoader(train_dataset , batch_size=batch_size, shuffle=True)

  val_dataset = CustomDataset(X_test_scaled, y_test)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader

