import torch.nn as nn
import torch
from abZono.zonotope import Zonotope
from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation

class ZonoOnnxBinaryMathOperation(nn.Module):
    
    def __init__(self, layer: OnnxBinaryMathOperation):
        super().__init__()
        self.broadcast = layer.broadcast
        self.axis = layer.axis
        self.math_op_function = layer.math_op_function
        if self.math_op_function == torch.add:
            self.math_op_function = Zonotope.__add__
        elif self.math_op_function == torch.sub:
            self.math_op_function = Zonotope.__sub__
        elif self.math_op_function == torch.mul:
            self.math_op_function = Zonotope.__mul__
        elif self.math_op_function == torch.div:
            self.math_op_function = Zonotope.__truediv__ # TODO: implement _onnx_div


    def forward(self, x: Zonotope, y: Zonotope):
        if self.broadcast == 1 and self.axis is not None:
            y = self.old_style_broadcast(x, y, self.axis)
        return self.math_op_function(x, y)

    def old_style_broadcast(self, x: Zonotope, y: Zonotope, axis: int):
        rank = len(x.shape)
        axis = axis + rank if axis < 0 else axis

        second_shape = [1] * axis + list(y.shape)
        second_shape = second_shape + [1] * (rank - len(second_shape))

        return y.view(second_shape)