import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from zonotope import Zonotope


class ZonoReLU(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x: Zonotope):
        l = x.center - x.generators.abs().sum()
        u = x.center + x.generators.abs().sum()
        if l > 0:
            return x
        elif u < 0:
            return Zonotope(torch.zeros_like(x.center), torch.zeros_like(x.generators))
        else:
            slope = u / (u - l)
            eta = u * (1 - slope) / 2
            return Zonotope(slope * x.center + eta, slope * x.generators)