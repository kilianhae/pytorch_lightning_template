"""
Define all metrics including your loss functions as classes with __callable__.
The main ingredients are parent classes that are specific to what the loss is used for e.g. measure accuracy or measure overfit, as they need to be called differently.
Then the select_loss function is used to return the correct loss given a string identifier.
The specific loss functions are defined as classes with __callable__.
"""
import torch
from torch import nn

class FidelityLoss():
    """
    A parent class for all losses that are used to measure prediction accuracy.
    """
    def __init__(self, loss_type: str, reduction: str = "mean"):
        self.reduction = reduction
        self.loss = select_loss(loss_type)
    def __call__(self, pred, out):
        return self.loss(pred, out)

class WeightDecayLoss():
    """
    A parent class for all losses that are used to get weight decay.
    """
    def __init__(self, loss_type = "L1"):
        super().__init__()
        self.loss_type = loss_type
        self.loss_func = select_loss(loss_type, reduction="sum")
    def __call__(self, net):
        loss = 0.0
        for name, param in net.named_parameters():
            if 'weight' in name:
                loss += self.loss_func(param, torch.zeros_like(param))
        return loss

class CustomLoss():
    """
	A specific custom loss function.
	"""
    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction)
    def __call__(self, pred, out):
        return self.loss(pred, out)

def select_loss(loss_type, reduction: str = "mean"):
    """
    Given a string identifier of the loss return the correct loss. Needs to support all implemented losses.
    """
    if loss_type == 'L1':
        return nn.L1Loss(reduction=reduction)
    elif loss_type == 'L2':
        return nn.MSELoss(reduction=reduction)
    elif loss_type == 'custom_loss':
        return CustomLoss(reduction=reduction)
    else:
        raise ValueError('loss_type should be L1 or L2 or custom_loss ...')
    

    

