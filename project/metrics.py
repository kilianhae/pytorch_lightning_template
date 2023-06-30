# define all metrics such as your loss but also your evaluation metrics here

from torch import nn


def MSELoss(pred, output):
    loss = nn.MSELoss()
    return loss(pred,output)

def L1Loss(pred, output):
    loss = nn.L1Loss()
    return loss(pred,output)
