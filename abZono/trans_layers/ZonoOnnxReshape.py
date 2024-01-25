import torch
import torch.nn as nn
from abZono.zonotope import Zonotope
from onnx2torch.node_converters.reshape import OnnxReshape


class ZonoOnnxReshape(nn.Module):

    def __init__(self, layer: OnnxReshape):
        super().__init__()
        self.__name__ = "ZonoOnnxReshape"

    def forward(self, x: Zonotope, shape):
        if torch.any(shape == 0):
            shape = [x.shape[i] if dim_size ==
                     0 else dim_size for i, dim_size in enumerate(shape)]
        return x.reshape(torch.Size(shape))
