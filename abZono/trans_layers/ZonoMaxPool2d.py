import torch.nn as nn
import torch.nn.functional as F
from zonotope import Zonotope


class ZonoMaxPool2d(nn.Module):

    def __init__(self, layer: nn.MaxPool2d):
        super().__init__()
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.ceil_mode = layer.ceil_mode

    def forward(self, x: Zonotope):
        return Zonotope(F.max_pool2d(x.center, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode),
                        F.max_pool2d(x.generators, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode))