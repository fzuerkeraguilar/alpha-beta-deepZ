import torch.nn as nn
from abZono.zonotope import Zonotope


class ZonoFlatten(nn.Module):

    def __init__(self, layer: nn.Flatten):
        super().__init__()
        self.start_dim = layer.start_dim
        self.end_dim = layer.end_dim
        self.__name__ = "ZonoFlatten"

    def forward(self, x: Zonotope):
        return x.flatten(self.start_dim, self.end_dim)
