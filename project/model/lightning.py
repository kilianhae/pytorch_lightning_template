# define the python wrappers for your pytorch models here, inlcuding the training loop
from project.model import Project_Model
from project.metrics import MSELoss, L1Loss

class PointMLP(Project_Model):
    def __init__(self, net, optimizer):
        super().__init__(net=net, optimizer=optimizer)
    
    def forward(self, x):
        return self.net(x)
    
    def f_step(self, batch, batch_idx, train):
        # if valid and train are very similar which they most likely are we dont have to change the functioanlity of the parent base class and can use f_step for both

        y = self.forward(batch[0])
        loss = MSELoss(y, batch[1])
        l_one_loss = L1Loss(y, batch[1])
        
        self.log_loss("MSELoss", loss, train)
        self.log_loss("L1Loss", l_one_loss, train)

        return y, loss

    


