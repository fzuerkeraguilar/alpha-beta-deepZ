import torch.nn as nn
from .trans_layers import ZonoConv, ZonoLinear, ZonoReLU


def transform_layer(layer: nn.Module):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv(layer)
    elif isinstance(layer, nn.Linear):
        return ZonoLinear(layer)
    elif isinstance(layer, nn.ReLU):
        return ZonoReLU(optimize_slope=False)
    else:
        raise NotImplementedError("Layer not supported: {}".format(layer))


def transform_network(network: nn.Module):
    layers = []
    for layer in network.children():
        layers.append(transform_layer(layer))
    return nn.Sequential(*layers)
