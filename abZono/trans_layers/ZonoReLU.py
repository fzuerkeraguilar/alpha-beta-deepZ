import torch
import torch.nn as nn
from abZono.zonotope import Zonotope


class ZonoReLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.__name__ = "ZonoReLU"

    def forward(self, x: Zonotope):
        gen_abs_sum = x.generators.abs().sum(dim=0)
        l = x.center - gen_abs_sum
        u = x.center + gen_abs_sum

        if torch.all(l > 0):
            return x
        elif torch.all(u < 0):
            return Zonotope(torch.zeros_like(x.center), torch.zeros_like(x.generators))
        else:
            where_crossing = torch.bitwise_and(l < 0, u > 0)
            where_smaller_zero = torch.bitwise_and(l < 0, u < 0)
            where_greater_zero = torch.bitwise_and(l > 0, u > 0)

            initial_slope = u / (u - l)  # slope with minimal area

            slope = initial_slope * where_crossing.float()
            new_generators = -slope * l * 0.5

            new_center = torch.where(where_crossing, x.center * slope + new_generators, x.center)
            new_center = torch.where(where_smaller_zero, torch.zeros_like(x.center), new_center)
            new_center = torch.where(where_greater_zero, x.center, new_center)

            return Zonotope(new_center, torch.cat((torch.where(where_crossing, x.generators * slope, x.generators), new_generators.unsqueeze(0))))


class ZonoAlphaReLU(nn.Module):

    def __init__(self, input_shape: torch.Size):
        super().__init__()
        self.slope_param = nn.Parameter(torch.ones(input_shape))
        self.__name__ = self.__class__.__name__

    def forward(self, x: Zonotope):
        l, u = x.l_u_bound
        if torch.all(l > 0):
            return x
        elif torch.all(u < 0):
            return Zonotope(torch.zeros_like(x.center), torch.zeros_like(x.generators))
        else:
            where_crossing = torch.bitwise_and(l < 0, u > 0)
            where_smaller_zero = torch.bitwise_and(l < 0, u < 0)
            where_greater_zero = torch.bitwise_and(l > 0, u > 0)

            slope = self.slope_param.clamp(0, 1)
            new_generators = - slope * l * 0.5 * where_crossing.float()

            new_center = torch.where(where_crossing, x.center * slope + new_generators, x.center)
            new_center = torch.where(where_smaller_zero, torch.zeros_like(x.center), new_center)
            new_center = torch.where(where_greater_zero, x.center, new_center)

            return Zonotope(new_center, torch.cat((torch.where(where_crossing, x.generators * slope, x.generators), new_generators.unsqueeze(0))))
