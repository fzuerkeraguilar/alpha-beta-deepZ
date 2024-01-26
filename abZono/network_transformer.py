import torch
import torch.nn as nn
import torch.fx
from torch.fx import Node
from typing import Dict
from abZono.trans_layers import ZonoConv2d, ZonoLinear, ZonoReLU, ZonoAlphaReLU, ZonoFlatten, ZonoOnnxConstant, \
    ZonoMaxPool2d, ZonoOnnxTranspose, ZonoOnnxReshape, ZonoOnnxPadStatic, ZonoOnnxPadDynamic, ZonoOnnxFlatten, \
    ZonoAvgPool1d, ZonoAvgPool2d, ZonoAvgPool3d, ZonoOnnxBinaryMathOperation
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.constant import OnnxConstant
from onnx2torch.node_converters.transpose import OnnxTranspose
from onnx2torch.node_converters.pad import OnnxPadStatic, OnnxPadDynamic
from onnx2torch.node_converters.flatten import OnnxFlatten
from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        shapes = {}
        for k, x in env.items():
            if isinstance(x, torch.Tensor):
                shapes[k] = x.shape
            elif isinstance(x, torch.fx.Node):
                shapes[k] = x.shape
            else:
                shapes[k] = None
        return shapes


def transform_layer(layer: nn.Module, input_shape, optimize_alpha=False, optimize_beta=False):
    if isinstance(layer, nn.Conv2d):
        return ZonoConv2d(layer)
    elif isinstance(layer, nn.Linear):
        return ZonoLinear(layer)
    elif isinstance(layer, nn.ReLU):
        if optimize_alpha:
            return ZonoAlphaReLU(input_shape)
        else:
            return ZonoReLU()
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


def transform_network_fx(network: torch.fx.GraphModule, input_tensor: torch.Tensor, optimize_alpha=False, optimize_beta=False):
    shapes = ShapeProp(network).propagate(input_tensor)
    new_modules = {}
    for node in network.graph.nodes:
        if node.op == 'call_module':
            replacement = transform_layer(network.get_submodule(node.target), shapes[node.name], optimize_alpha,
                                          optimize_beta)
            if replacement is not None:
                zono_target = f"{node.target}_zono"
                new_modules[zono_target] = replacement
                node.target = zono_target
            else:
                raise Exception(f"Could not transform {node.target}")

    for name, module in new_modules.items():
        network.add_module(name, module)

    network.delete_all_unused_submodules()
    network.graph.lint()
    network.recompile()
    return network
