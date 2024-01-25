import re
import torch.nn as nn
from onnx2torch.node_converters.pad import OnnxPadStatic, OnnxPadDynamic
from abZono.zonotope import Zonotope


class ZonoOnnxPadStatic(nn.Module):
    
        def __init__(self, layer: OnnxPadStatic):
            super().__init__()
            self.mode = layer.mode
            self.value = layer.constant_value
            self.pads = layer.pads
            self.__name__ = "ZonoOnnxPadStatic"

        def forward(self, x: Zonotope):
              return x.pad(self.pads, self.mode, self.value)
        
class ZonoOnnxPadDynamic(nn.Module):
      
        def __init__(self, layer: OnnxPadDynamic):
            super().__init__()
            self.mode = layer.mode
            self.value = layer.value
            self.pads = layer.pads
            self.__name__ = "ZonoOnnxPadDynamic"

        def forward(self, x: Zonotope):
              return x.pad(self.pads, self.mode, self.value)