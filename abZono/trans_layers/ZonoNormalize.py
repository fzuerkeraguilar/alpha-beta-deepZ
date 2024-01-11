import torch.nn as nn
import torch.nn.functional as F
from zonotope import Zonotope


class ZonoNormalize(nn.Module):
    def __init__(self, layer: nn.Module):
        if hasattr(layer, "mean"):
            self.mean = layer.mean
        else:
            self.mean = 0.1307

        if hasattr(layer, "std"):
            self.std = layer.std
        else:
            self.std = 0.3081
        super().__init__()

    def forward(self, x: Zonotope):
        return Zonotope((x.center - self.mean) / self.std, x.generators / self.std)