import torch.nn as nn
from .trans_layers import ZonoConv, ZonoLinear, ZonoReLU


def transform_layer(layer: nn.Module, optimize_alpha=False, optimize_beta=False):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv(layer)
    elif isinstance(layer, nn.Linear):
        return ZonoLinear(layer)
    elif isinstance(layer, nn.ReLU):
        return ZonoReLU(optimize_slope=optimize_alpha)
    else:
        raise NotImplementedError("Layer not supported: {}".format(layer))


def transform_network(network: nn.Module, optimize_alpha=False, optimize_beta=False):
    layers = []
    for layer in network.children():
        layers.append(transform_layer(layer, optimize_alpha, optimize_beta))
    return nn.Sequential(*layers)
