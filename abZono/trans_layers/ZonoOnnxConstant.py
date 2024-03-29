import torch.nn as nn
from onnx2torch.node_converters.constant import OnnxConstant


class ZonoOnnxConstant(nn.Module):

    def __init__(self, layer: OnnxConstant):
        super().__init__()
        self.value = layer.value
        self.__name__ = "ZonoOnnxConstant"

    def forward(self, *args, **kwargs):
        return self.value
