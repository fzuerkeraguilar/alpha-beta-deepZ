import torch
import torch.nn as nn
from ..zonotope import Zonotope


class ZonoReLU(nn.Module):

    def __init__(self, optimize_slope=False):
        super().__init__()
        self.optimize_slope = optimize_slope

    def forward(self, x: Zonotope):
        l = x.center - x.generators.abs().sum(dim=0)
        u = x.center + x.generators.abs().sum(dim=0)
        if torch.all(l > 0):
            return x
        elif torch.all(u < 0):
            return Zonotope(torch.zeros_like(x.center), torch.zeros_like(x.generators))
        else:
            where_crossing = torch.bitwise_and(l < 0, u > 0)
            initial_slope = u / (u - l)
            if self.optimize_slope:
                self.slope.data = torch.clamp(self.slope.data, 0, initial_slope)  # clamp the slope
                slope = self.slope * where_crossing.float()  # apply the mask to the slope
            else:
                slope = initial_slope * where_crossing.float()
            new_generators = -slope * l * 0.5
            return Zonotope(torch.where(where_crossing, x.center * slope + new_generators, x.center),
                            torch.cat((torch.where(where_crossing, x.generators * slope, x.generators),
                                       new_generators.unsqueeze(0))))
