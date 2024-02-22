import torch
import torch.nn as nn
import torch.nn.functional as F
from abZono.zonotope import Zonotope


class ZonoConv2d(nn.Module):

    def __init__(self, layer: nn.Conv2d):
        super().__init__()
        self.register_buffer("weight", layer.weight.data)
        if layer.bias is not None:
            self.register_buffer("bias", layer.bias.data)
        else:
            self.bias = None
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups
        self.__name__ = self.__class__.__name__

    def forward(self, x: Zonotope):
        conv_center = F.conv2d(x.center, self.weight, self.bias,
                               self.stride, self.padding, self.dilation, self.groups)
        # Prepare the generators for batched convolution
        # Flatten the generator dimension into the batch dimension
        num_generators, batch_size, C, H, W = x.generators.shape
        generators_flat = x.generators.view(batch_size * num_generators, C, H, W)

        # Convolve generators without grouping, as each generator is treated independently
        conv_generators = F.conv2d(generators_flat, self.weight, None,  # No bias for generators
                                   self.stride, self.padding, self.dilation, 1)  # Use groups=1 for individual handling

        # Reshape the convolved generators back to the original format
        _, C_out, H_out, W_out = conv_generators.shape
        conv_generators = conv_generators.view(num_generators, batch_size, C_out, H_out, W_out)

        return Zonotope(conv_center, conv_generators)
