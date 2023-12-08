import torch.nn as nn
import torch.nn.functional as F
from ..zonotope import Zonotope


class ZonoLinear(nn.Module):

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data if layer.bias is not None else None

    def forward(self, x: Zonotope):
        if self.bias is not None:
            return Zonotope(F.linear(x.center, self.weight, self.bias), F.linear(x.generators, self.weight, self.bias))
        else:
            return Zonotope(F.linear(x.center, self.weight), F.linear(x.generators, self.weight))
