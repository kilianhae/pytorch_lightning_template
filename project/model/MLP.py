from torch import nn


class MLP(nn.Module):
    def __init__(self, inputdim, emb_dim=100, num_layer=5, **kwargs):
        super(MLP, self).__init__()

        self.hidden = nn.ModuleList(
            [nn.Sequential(nn.Linear(inputdim, emb_dim), nn.ReLU())]
        )
        for i in range(num_layer - 1):
            self.hidden.append(nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU()))
        self.hidden.append(nn.Linear(emb_dim, 1))

    def forward(self, x):
        for seq in self.hidden:
            x = seq(x)
        return x
