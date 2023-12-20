import torch.nn as nn
import torch.nn.functional as F
from ..zonotope import Zonotope
from ..functions.ZonoConvFun import ZonoConvFun


class ZonoConv:

    def __init__(self, layer: nn.Conv2d):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups

    def forward(self, x: Zonotope):
        return ZonoConvFun.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
