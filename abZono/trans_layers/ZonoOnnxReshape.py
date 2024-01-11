import torch
import torch.nn as nn
from zonotope import Zonotope
from onnx2torch.node_converters.reshape import OnnxReshape

class ZonoOnnxReshape(nn.Module):

    def __init__(self, layer: OnnxReshape):
        super().__init__()
        self.shape = layer.shape

    def forward(self, x: Zonotope):
        return Zonotope(x.center.reshape(self.shape), x.generators.reshape(self.shape))