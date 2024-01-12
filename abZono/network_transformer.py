import torch.nn as nn
from trans_layers import ZonoConv, ZonoLinear, ZonoReLU, ZonoFlatten, ZonoOnnxConstant, ZonoMaxPool2d, ZonoOnnxTranspose, ZonoOnnxReshape
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.constant import OnnxConstant
from onnx2torch.node_converters import OnnxTranspose


def transform_layer(layer: nn.Module, optimize_alpha=False, optimize_beta=False):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv(layer)
    elif isinstance(layer, nn.Linear):
        return ZonoLinear(layer)
    elif isinstance(layer, nn.ReLU):
        return ZonoReLU(optimize_slope=optimize_alpha)
    elif isinstance(layer, nn.Flatten):
        return ZonoFlatten(layer)
    elif isinstance(layer, nn.MaxPool2d):
        return ZonoMaxPool2d(layer)
    elif isinstance(layer, OnnxReshape):
        return ZonoOnnxReshape(layer)
    elif isinstance(layer, OnnxTranspose):
        return ZonoOnnxTranspose(layer)
    elif isinstance(layer, OnnxConstant):
        return ZonoOnnxConstant(layer)
    else:
        return layer


def transform_network(network: nn.Module, optimize_alpha=False, optimize_beta=False):
    for name, module in network.named_children():
        replacement = transform_network(module, optimize_alpha, optimize_beta)
        if replacement is not None:
            setattr(network, name, replacement)
    return transform_layer(network, optimize_alpha, optimize_beta)
