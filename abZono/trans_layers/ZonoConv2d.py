import torch
import torch.nn as nn
import torch.nn.functional as F
from abZono.zonotope import Zonotope


class ZonoConv2d(nn.Module):

    def __init__(self, layer: nn.Conv2d):
        super().__init__()
        self.weight = layer.weight.data
        self.bias = layer.bias.data if layer.bias is not None else None
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups
        self.__name__ = 'ZonoConv2d'

    def forward(self, x: Zonotope):
        conv_center = F.conv2d(x.center, self.weight, self.bias,
                               self.stride, self.padding, self.dilation, self.groups)
        conv_generators = torch.stack([F.conv2d(
            gen, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups) for gen in x.generators])
        return Zonotope(conv_center, conv_generators)
