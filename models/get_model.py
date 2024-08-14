from models.binary_classifiers import *
from models.multiclass_classifier import *
from models.autoencoder import *
from models.residual_autoencoder import *
def get_model(**kwargs):
    '''
    Initialize and return the model based on the specified type.

    Parameters:
    - **kwargs: Additional keyword arguments for model initialization.
      - model_type (str): The type of model to initialize ('BinaryClassifier' or 'BinaryClassifier_v2').
      - input_dim (int): The input dimension for the model.

    Returns:
    - model (nn.Module): The initialized model.
    '''
    model_type = kwargs.get('model_type')
    input_dim = kwargs.get('input_dim')

    if model_type == 'BinaryClassifier':
        print(' *** loading ', model_type, '***')
        return BinaryClassifier(input_dim)
    elif model_type == 'BinaryClassifier_v2':
        print(' *** loading ', model_type, '***')
        return BinaryClassifier_v2(input_dim)
    elif model_type == 'QuarkGluonClassifierWithEmbeddings':
        print(' *** loading ', model_type, '***')
        return QuarkGluonClassifierWithEmbeddings(num_particles, embedding_dim, jet_input_dim, hidden_dim, output_dim)
    elif model_type == 'MulticlassClassifier':
        print(' *** loading ', model_type, '***')
        return MulticlassClassifier(**kwargs)
    elif model_type == 'MulticlassClassifier_v2':
        print(' *** loading ', model_type, '***')
        return MulticlassClassifier_v2(**kwargs)
    elif model_type == 'autoencoder':
        print(' *** loading ', model_type, '***')
        return TabularAutoencoder(**kwargs)
    elif model_type == 'residual_autoencoder':
        print(' *** loading ', model_type, '***')
        return autoencoder(**kwargs)
    
    else:
        raise ValueError("Invalid model type specified. Choose 'BinaryClassifier' or 'BinaryClassifier_v2'.")
