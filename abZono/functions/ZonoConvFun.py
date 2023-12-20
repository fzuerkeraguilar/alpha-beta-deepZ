from torch.autograd import Function
from ..zonotope import Zonotope

class ZonoConvFun(Function):

    @staticmethod
    def forward(ctx, zonotope: Zonotope, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(zonotope, weight, bias, stride, padding, dilation, groups)
        # Your forward computation here
        return zonotope.conv2d(weight, bias, stride, padding, dilation, groups)
    
    @staticmethod
    def backward(ctx, grad_output):
        zonotope, weight, bias, stride, padding, dilation, groups = ctx.saved_tensors
        grad_zonotope = None
        # Your backward computation here
        # This should compute grad_zonotope = dL/dzonotope, where L is the loss
        return grad_zonotope, None, None, None, None, None, None