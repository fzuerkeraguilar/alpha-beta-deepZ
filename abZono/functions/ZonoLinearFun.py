from torch.autograd import Function
from ..zonotope import Zonotope


class ZonoLinearFun(Function):

    @staticmethod
    def forward(ctx, zonotope: Zonotope, weight, bias):
        ctx.save_for_backward(zonotope, weight, bias)
        # Your forward computation here
        return zonotope.linear(weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        zonotope, weight, bias = ctx.saved_tensors
        grad_zonotope = None
        # Your backward computation here
        # This should compute grad_zonotope = dL/dzonotope, where L is the loss
        return grad_zonotope, None, None
