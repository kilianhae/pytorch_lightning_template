# here define the base functionality that all models have and adhere to
import lightning.pytorch as pl

from project.optim import get_optimizer

class Project_Model(pl.LightningModule):
    def __init__(self, net, optimizer):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
    
    def training_step(self, batch, batch_idx):
        _, loss = self.f_step(batch, batch_idx, train = True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self.f_step(batch, batch_idx, train = False)
        return pred
    
    # def predict_step(self, batch, batch_idx):
    #     pred = self.forward(batch, batch_idx)
    #     return pred

    def log_loss(self, loss_name, loss, train, prog_bar = True):
    # this is a function that is implemented such that we dont have to expelicitly write different metric name for val and train metrics but can just tag with train True or False
    # auto logs everything on epoch and on step level
        if train:
            self.log("train_" + loss_name, loss, prog_bar=prog_bar)
            self.log("train_epoch_" + loss_name, loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        else:
            self.log("val_" + loss_name, loss, prog_bar=prog_bar)
            self.log("val_epoch_" + loss_name, loss, prog_bar=prog_bar, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return get_optimizer(self.net.parameters(), self.optimizer)