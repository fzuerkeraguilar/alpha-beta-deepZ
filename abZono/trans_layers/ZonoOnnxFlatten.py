import re
import torch.nn as nn
from abZono.zonotope import Zonotope
from onnx2torch.node_converters.flatten import OnnxFlatten


class ZonoOnnxFlatten(nn.Module):

    def __init__(self, layer: OnnxFlatten):
        super().__init__()
        self.axis = layer.axis
        self.__name__ = "ZonoOnnxFlatten"

    def forward(self, x: Zonotope):
        x = x.flatten(end_dim=self.axis - 1)
        return x.flatten(start_dim=1)
