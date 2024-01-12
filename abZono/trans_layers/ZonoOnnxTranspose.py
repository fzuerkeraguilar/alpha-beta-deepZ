import torch
import torch.nn as nn
from zonotope import Zonotope
from onnx2torch.node_converters.transpose import OnnxTranspose


class ZonoOnnxTranspose(nn.Module):

    def __init__(self, layer: OnnxTranspose):
        super().__init__()
        self.perm = layer.perm

    def forward(self, x: Zonotope):
        center_perm = [self.perm[i + 1] - 1 for i in range(len(x.center.shape))]
        return Zonotope(x.center.permute(*center_perm), x.generators.permute(*self.perm))
