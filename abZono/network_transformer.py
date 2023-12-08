import torch.nn as nn
from .trans_layers import ZonoConv, ZonoLinear, ZonoReLU


def trans_layer(layer: nn.Module):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv(layer)
    elif isinstance(layer, nn.Linear):
        return ZonoLinear(layer)
    elif isinstance(layer, nn.ReLU):
        return ZonoReLU()
    else:
        raise Exception("Unknown layer type")


def trans_network(layers):
    return nn.Sequential(*layers)
