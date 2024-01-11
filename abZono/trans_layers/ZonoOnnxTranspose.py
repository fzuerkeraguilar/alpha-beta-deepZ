import torch
import torch.nn as nn
from zonotope import Zonotope
from onnx2torch.node_converters.transpose import OnnxTranspose

class ZonoOnnxTranspose(nn.Module):
    
        def __init__(self, layer: OnnxTranspose):
            super().__init__()
            self.perm = layer.perm
    
        def forward(self, x: Zonotope):
            return Zonotope(x.center.permute(*self.perm), x.generators.permute(*self.perm))