import torch
import torch.nn as nn
import torch.nn.functional as F
from zonotope import Zonotope


class ZonoMaxPool2d(nn.Module):

    def __init__(self, layer: nn.MaxPool2d):
        super().__init__()
        # Ensure that kernel_size, stride, padding, and dilation are in the correct format (int or tuple)
        self.kernel_size = self._to_tuple(layer.kernel_size)
        if layer.stride is not None:
            self.stride = self._to_tuple(layer.stride)
        else:
            self.stride = None
        self.padding = self._to_tuple(layer.padding)
        self.dilation = self._to_tuple(layer.dilation)
        self.ceil_mode = layer.ceil_mode

    def forward(self, x: Zonotope):
        # Apply max_pool2d to the center and each generator
        pooled_center = F.max_pool2d(
            x.center, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
        pooled_generators = torch.stack([F.max_pool2d(
            gen, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode) for gen in x.generators])
        return Zonotope(pooled_center, pooled_generators)

    @staticmethod
    def _to_tuple(x):
        if isinstance(x, int):
            return (x, x)
        return x
