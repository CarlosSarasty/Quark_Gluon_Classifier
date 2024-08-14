def get_loss(func, **params):
  if func == 'focal_loss':
    from loss.loss import Focal_loss
    return Focal_loss

  else:
    from torch import nn
    return getattr(nn.modules.loss, func)(**params)
