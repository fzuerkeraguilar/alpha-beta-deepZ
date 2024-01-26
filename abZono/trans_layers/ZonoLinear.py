import torch.nn as nn
import torch.nn.functional as F
from abZono.zonotope import Zonotope


class ZonoLinear(nn.Module):

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.register_buffer("weight", layer.weight.data)
        if layer.bias is not None:
            self.register_buffer("bias", layer.bias.data)
        else:
            self.bias = None
        self.__name__ = self.__class__.__name__

    def forward(self, x: Zonotope):
        return Zonotope(F.linear(x.center, self.weight, self.bias),
                        F.linear(x.generators, self.weight))
