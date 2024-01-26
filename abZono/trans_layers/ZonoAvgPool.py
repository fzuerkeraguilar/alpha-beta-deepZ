import torch
import torch.nn as nn
import torch.nn.functional as F
from abZono.zonotope import Zonotope


class ZonoAvgPool1d(nn.Module):

    def __init__(self, layer: nn.AvgPool1d):
        super().__init__()
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.ceil_mode = layer.ceil_mode
        self.count_include_pad = layer.count_include_pad
        self.__name__ = 'ZonoAvgPool1d'

    def forward(self, x: Zonotope):
        pooled_center = F.avg_pool1d(x.center, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        pooled_generators = torch.stack([F.avg_pool1d(
            gen, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad) for gen in
            x.generators])
        return Zonotope(pooled_center, pooled_generators)


class ZonoAvgPool2d(nn.Module):

    def __init__(self, layer: nn.AvgPool2d):
        super().__init__()
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.ceil_mode = layer.ceil_mode
        self.count_include_pad = layer.count_include_pad
        self.__name__ = 'ZonoAvgPool2d'

    def forward(self, x: Zonotope):
        pooled_center = F.avg_pool2d(x.center, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        pooled_generators = torch.stack([F.avg_pool2d(
            gen, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad) for gen in
            x.generators])
        return Zonotope(pooled_center, pooled_generators)


class ZonoAvgPool3d(nn.Module):

    def __init__(self, layer: nn.AvgPool3d):
        super().__init__()
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.ceil_mode = layer.ceil_mode
        self.count_include_pad = layer.count_include_pad
        self.__name__ = 'ZonoAvgPool3d'

    def forward(self, x: Zonotope):
        pooled_center = F.avg_pool3d(x.center, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        pooled_generators = torch.stack([F.avg_pool3d(
            gen, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad) for gen in
            x.generators])
        return Zonotope(pooled_center, pooled_generators)
