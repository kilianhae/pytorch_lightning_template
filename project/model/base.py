"""
Defines the base LightningModule that all models inherit from
"""

import lightning.pytorch as pl
from project.optim import get_optimizer


class Project_Model(pl.LightningModule):
    """PROJECT model.

    Kind of abstract class for the lightning modules used in the project package.

    We define the training and validation steps, and the optimizer. This is useful for uniformity.

    Parameters
    ----------
    net: nn.Module
        Pytorch Neural network
    optimizer: torch.optim.Optimizer
        Optimizer to use.

    Methods
    -------
    training_step(batch, batch_idx)
        Training step. Calls f_step which runs forward, logs and returns the loss.
    validation_step(batch, batch_idx)
        Validation step. Calls f_step which runs forward, logs and returns the prediction.
    log_loss(loss_name, loss, train, prog_bar = True)
        Called to log the metrics on step and epoch level invariant to training or validation.
    configure_optimizers()
        Returns the optimizer.
    """

    def __init__(self, net, optimizer, lr):
        """Initialize the network and the optimizer string."""
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.lr = lr

    def training_step(self, batch, batch_idx):
        """Calls f_step which runs forward, logs and returns the loss."""
        _, loss = self.f_step(batch, batch_idx, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Calls f_step which runs forward, logs and returns the prediction."""
        pred, loss = self.f_step(batch, batch_idx, train=False)
        return pred

    # def predict_step(self, batch, batch_idx):

    def log_loss(self, loss_name, loss, train, prog_bar=True):
        """
        This is a function that is implemented such that we dont have to expelicitly write different metric name for val and train metrics but can just tag with train True or False

        Parameters
        ----------
        loss_name : str
            Name of the loss to log.
        loss : torch.Tensor
            Loss to log.
        train : bool
            Whether we are in training or validation mode.
        prog_bar : bool, optional
            Whether to log to the progress bar. The default is True.
        """
        if train:
            self.log("train_" + loss_name, loss, prog_bar=prog_bar)
            self.log(
                "train_epoch_" + loss_name,
                loss,
                prog_bar=prog_bar,
                on_epoch=True,
                on_step=False,
            )
        else:
            self.log(
                "val_" + loss_name,
                loss,
                prog_bar=prog_bar,
                on_epoch=False,
                on_step=True,
            )
            self.log(
                "val_epoch_" + loss_name,
                loss,
                prog_bar=prog_bar,
                on_epoch=True,
                on_step=False,
            )

    def configure_optimizers(self, *args, **kwargs):
        """Returns the optimizer."""
        return get_optimizer(self.net.parameters(), self.optimizer, self.lr)
