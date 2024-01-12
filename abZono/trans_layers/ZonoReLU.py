import torch
import torch.nn as nn
import torch.nn.functional as F
from zonotope import Zonotope


class ZonoReLU(nn.Module):

    def __init__(self, optimize_slope=False):
        super().__init__()
        self.slope = None
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
            initial_slope = u / (u - l)  # slope with minimal area
            if self.optimize_slope:
                self.slope_param = nn.Parameter(torch.ones_like(initial_slope))
                self.slope = torch.sigmoid(self.slope_param) * initial_slope
                new_generators = (torch.ones_like(self.slope) - self.slope) * u * 0.5 * where_crossing.float()
                return Zonotope(torch.where(where_crossing, x.center * self.slope + new_generators, x.center),
                                torch.cat((torch.where(where_crossing, x.generators * self.slope, x.generators),
                                        new_generators)))
            else:
                self.slope = initial_slope * where_crossing.float()
                new_generators = -self.slope * l * 0.5
                if new_generators.dim() < x.generators.dim():
                    new_generators_unsqueezed = new_generators.unsqueeze(0)
                else:
                    new_generators_unsqueezed = new_generators
                return Zonotope(torch.where(where_crossing, x.center * self.slope + new_generators, x.center),
                                torch.cat((torch.where(where_crossing, x.generators * self.slope, x.generators),
                                        new_generators_unsqueezed)))
