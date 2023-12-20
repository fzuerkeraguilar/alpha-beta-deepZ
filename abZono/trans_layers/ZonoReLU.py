import torch.nn as nn
from ..zonotope import Zonotope
from ..functions.ZonoReLUFun import ZonoReLUFun

class ZonoReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Zonotope):
        return  ZonoReLUFun.apply(x)
