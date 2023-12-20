import torch.nn as nn
from ..zonotope import Zonotope

class ZonoReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Zonotope):
        return x.relu()
