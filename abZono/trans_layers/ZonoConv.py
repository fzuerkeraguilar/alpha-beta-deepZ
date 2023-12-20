import torch.nn as nn
import torch.nn.functional as F
from ..zonotope import Zonotope


class ZonoConv:

    def __init__(self, layer: nn.Conv2d):
        super().__init__()
        self.weight = layer.weight.data
        self.weight.requires_grad = False
        self.bias = layer.bias.data
        self.bias.requires_grad = False
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups

    def forward(self, x: Zonotope):
        return Zonotope(F.conv2d(x.center, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups),
                        F.conv2d(x.generators, self.weight, None, self.stride, self.padding, self.dilation, self.groups))