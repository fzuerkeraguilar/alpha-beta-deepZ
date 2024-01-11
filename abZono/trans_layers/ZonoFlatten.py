import torch.nn as nn
import torch
from zonotope import Zonotope

class ZonoFlatten:

    def __init__(self, layer: nn.Flatten):
        super().__init__()
        self.start_dim = layer.start_dim
        self.end_dim = layer.end_dim

    def forward(self, x: Zonotope):
        return Zonotope(torch.flatten(x.center, self.start_dim, self.end_dim), torch.flatten(x.generators, self.start_dim, self.end_dim))
