import torch

def emd2_loss(pred, true, reduction='mean'):
    """
    Calculate the EMD^2 loss between the true labels and predicted probabilities.

    Args:
    - pred (torch.Tensor): A tensor of shape (N, C) representing the predicted probabilities 
                        for each class. N is the number of events, and C is the number of classes.
    - true (torch.Tensor): A tensor of shape (N, C) representing the true labels in one-hot 
                        encoded format. N is the number of events, and C is the number of classes.
    - reduction (str, optional):  Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                        'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 
                        'sum': the output will be summed.

    Returns:
    - emd2_loss (float): The EMD^2 loss.
    """
    # Calculate the cumulative distribution functions (CDFs) for both true labels and predictions
    _ct = torch.cumsum(true, dim=1)
    _cp = torch.cumsum(pred, dim=1)

    # Calculate the squared difference between the true and predicted CDFs
    squared_diff = (_ct - _cp) ** 2

    # Sum over classes and take the mean over all events
    if  reduction == 'mean': 
        emd2_loss = torch.sum(squared_diff, dim=1).mean()
    else:
        emd2_loss = torch.sum(squared_diff, dim=1)
    return emd2_loss
