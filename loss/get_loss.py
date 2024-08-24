def get_loss(**params):
   
    func = params.get('loss_function')

    if func == 'emd2_loss':
        from loss.loss import emd2_loss
        return emd2_loss
  
    if func == 'focal_loss':
        from loss.focal_loss import Focal_loss
        return Focal_loss

    else:
        from torch import nn
        return getattr(nn.modules.loss, func)(**params)
