import torch 
import torch.nn as nn 

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.seq(x)
    

class Projector(nn.Module):
    def __init__(self, proj_dim=32):
        super(Projector, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(30, 128, bias=False),
            nn.LeakyReLU(),
            nn.Linear(128, proj_dim, bias=False)
        )
    def forward(self, x):
        return self.seq(x)