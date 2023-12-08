import torch
import torch.nn as nn


class ExposeNet(nn.Module):
    def __init__(self):
        super(ExposeNet, self).__init__()
        self.layer1 = nn.Linear(2, 2, bias=False)
        self.layer1.weight.data = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], dtype=torch.float)
        self.layer2 = nn.ReLU()
        self.layers = [self.layer1, self.layer2]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
