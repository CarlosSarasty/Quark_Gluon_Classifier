import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor in range (0,1) to balance the importance of positive vs negative examples.
        :param gamma: Focusing parameter for modulating factor (1-p).
        :param reduction: 'none' | 'mean' | 'sum'. Specifies the reduction to apply to the output.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute the pt factor, which is just the probability of the target class
        pt = torch.exp(-BCE_loss)

        # Compute the focal loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
