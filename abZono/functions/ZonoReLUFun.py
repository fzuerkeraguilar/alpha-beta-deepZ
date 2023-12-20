from torch.autograd import Function
from ..zonotope import Zonotope

class ZonotopeReLU(Function):
    @staticmethod
    def forward(ctx, zonotope: Zonotope):
        ctx.save_for_backward(zonotope)
        # Your forward computation here
        return zonotope.relu()

    @staticmethod
    def backward(ctx, grad_output):
        zonotope, = ctx.saved_tensors
        grad_zonotope = None
        # Your backward computation here
        # This should compute grad_zonotope = dL/dzonotope, where L is the loss
        return grad_zonotope