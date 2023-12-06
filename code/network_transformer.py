import torch.nn as nn
from trans_layers.Conv import ZonoConv
from trans_layers.Linear import ZonoLinear
from trans_layers.ReLU import ZonoReLU



def trans_layer(layer: nn.Module):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv(layer)
    elif isinstance(layer, nn.Linear):
        return ZonoLinear(layer)
    elif isinstance(layer, nn.ReLU):
        return ZonoReLU()
    else:
        raise Exception("Unknown layer type")