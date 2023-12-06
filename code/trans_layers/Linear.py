import torch.nn as nn
import torch.nn.functional as F
from zonotope import Zonotope


class ZonoLinear(nn.Module):

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data

    def forward(self, x:Zonotope):
        y = F.linear(x, self.weight, self.bias)
        return y
