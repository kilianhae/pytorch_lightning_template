from  torch import nn

class MLP(nn.Module):
    def __init__(self, inputdim, emb_dim=100, num_layer=5, **kwargs):
        self.center=True
        super(MLP, self).__init__()
        # num layers+1 layers since we have a start and end layer and layers-1 hidden ones
        self.hidden = nn.ModuleList([nn.Sequential(nn.Linear(inputdim,emb_dim),nn.ReLU())])
        for i in range(num_layer-1):
            self.hidden.append(nn.Sequential(nn.Linear(emb_dim,emb_dim),nn.ReLU()))
        self.hidden.append(nn.Linear(emb_dim,1))

    def forward(self, x):
        for seq in self.hidden:
            x=seq(x)
        return x