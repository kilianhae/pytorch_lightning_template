# define the python wrappers for your pytorch models here, inlcuding the training loop
from project.model import Project_Model
from project.metrics import FidelityLoss, WeightDecayLoss


class PointMLP(Project_Model):
    """A lightning module specific to a modelling approach.

    Parameters
    ----------
    net: nn.Module
        Pytorch Neural network
    optimizer: torch.optim.Optimizer
        Optimizer to use.
    loss_type: str
        Type of loss to use.
    weight_decay: float (optional)
        Weight decay to use (if to use it).

    Methods
    -------
    forward(x)
        Forward pass of the pytorch network.
    f_step(batch, batch_idx, train)
        A function that is shared between train and validation and runs forward and logs the wanted metrics.
        This can be replaced with explicit train and validation steps if needed when they dont share the same functionality.
        Is called by training_step and validation_step.
    """

    def __init__(self, net, optimizer: str, loss_type: str, lr: float, weight_decay: float = 0,):
        super().__init__(net=net, optimizer=optimizer, lr=lr)
        self.loss_type = loss_type
        self.loss = FidelityLoss(loss_type)
        self.weight_decay = weight_decay

        if weight_decay > 0:
            self.wd_loss = WeightDecayLoss(loss_type)

    def forward(self, x):
        """Forward pass of the pytorch network. Used for prediction step."""
        return self.net(x)

    def f_step(self, batch, batch_idx, train):
        """
        if valid and train are very similar which they most likely are we dont have to change the functioanlity of the parent base class and can use f_step for both
        """
        y = self.forward(batch[0])
        loss = self.loss(y, batch[1])
        self.log_loss("loss", loss, train)
        print(loss)
        if self.weight_decay > 0:
            wd_loss = self.wd_loss(self.net)
            self.log_loss("wd_loss", wd_loss, train)
            loss = loss + wd_loss * self.weight_decay

        return y, loss
