import torch
import torch.nn as nn
from abZono.zonotope import Zonotope
from onnx2torch.node_converters.transpose import OnnxTranspose


class ZonoOnnxTranspose(nn.Module):

    def __init__(self, layer: OnnxTranspose):
        super().__init__()
        self.perm = layer.perm
        self.__name__ = "ZonoOnnxTranspose"

    def forward(self, x: Zonotope):
        if self.perm is None:
            self.perm = list(range(x.center.dim()))[::-1]
        return x.permute(self.perm)
