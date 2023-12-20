import torch.nn as nn
from ..zonotope import Zonotope
from ..functions.ZonoLinearFun import ZonoLinearFun


class ZonoLinear(nn.Module):

    def __init__(self, layer: nn.Linear):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data if layer.bias is not None else None

    def forward(self, x: Zonotope):
        return ZonoLinearFun.apply(x, self.weight, self.bias)
