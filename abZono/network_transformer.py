import torch.nn as nn
from trans_layers import ZonoConv2d, ZonoLinear, ZonoReLU, ZonoFlatten, ZonoOnnxConstant, ZonoMaxPool2d, ZonoOnnxTranspose, ZonoOnnxReshape, ZonoOnnxPadStatic, ZonoOnnxPadDynamic, ZonoOnnxFlatten, ZonoAvgPool1d, ZonoAvgPool2d, ZonoAvgPool3d, ZonoOnnxBinaryMathOperation
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.constant import OnnxConstant
from onnx2torch.node_converters.transpose import OnnxTranspose
from onnx2torch.node_converters.pad import OnnxPadStatic, OnnxPadDynamic
from onnx2torch.node_converters.flatten import OnnxFlatten
from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation


def transform_layer(layer: nn.Module, optimize_alpha=False, optimize_beta=False):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv2d(layer)
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
    elif isinstance(layer, OnnxPadStatic):
        return ZonoOnnxPadStatic(layer)
    elif isinstance(layer, OnnxPadDynamic):
        return ZonoOnnxPadDynamic(layer)
    elif isinstance(layer, OnnxFlatten):
        return ZonoOnnxFlatten(layer)
    elif isinstance(layer, nn.AvgPool1d):
        return ZonoAvgPool1d(layer)
    elif isinstance(layer, nn.AvgPool2d):
        return ZonoAvgPool2d(layer)
    elif isinstance(layer, nn.AvgPool3d):
        return ZonoAvgPool3d(layer)
    elif isinstance(layer, OnnxBinaryMathOperation):
        return ZonoOnnxBinaryMathOperation(layer)
    else:
        return layer


def transform_network(network: nn.Module, optimize_alpha=False, optimize_beta=False):
    for name, module in network.named_children():
        replacement = transform_network(module, optimize_alpha, optimize_beta)
        if replacement is not None:
            setattr(network, name, replacement)
    return transform_layer(network, optimize_alpha, optimize_beta)
