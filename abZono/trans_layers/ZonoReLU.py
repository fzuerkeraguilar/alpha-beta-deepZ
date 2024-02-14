import torch
import torch.nn as nn
from abZono.zonotope import Zonotope


class ZonoReLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.__name__ = "ZonoReLU"

    def forward(self, x: Zonotope):
        l, u = x.l_u_bound
        zero_tensor = torch.zeros_like(x.center)
        negative_mask = (u < 0)
        crossing_mask = (l < 0) & (u > 0)

        slope = u / (u - l)
        new_generator = -slope * l * 0.5 * crossing_mask.float()

        new_center = torch.where(crossing_mask, x.center * slope + new_generator, x.center)
        new_center = torch.where(negative_mask, zero_tensor, new_center)

        compressed_generator_indices = crossing_mask.nonzero(as_tuple=True)
        num_activations = compressed_generator_indices[0].size(0)
        do_not_repeat_other_dims = [1] * x.center.dim()
        stacked_generator_indices = (torch.arange(num_activations), *compressed_generator_indices)

        new_eps_terms = zero_tensor.unsqueeze(0).repeat(num_activations, *do_not_repeat_other_dims)
        new_eps_terms[stacked_generator_indices] = new_generator[compressed_generator_indices]

        old_new_generators = torch.where(crossing_mask, x.generators * slope, x.generators)
        old_new_generators = torch.where(negative_mask, torch.zeros_like(x.generators), old_new_generators)

        return Zonotope(new_center, torch.cat((old_new_generators, new_eps_terms), dim=0))


class ZonoAlphaReLU(nn.Module):

    def __init__(self, input_shape: torch.Size):
        super().__init__()
        self.slope_param = nn.Parameter(torch.rand(input_shape))
        self.first_run = True
        self.__name__ = self.__class__.__name__

    def forward(self, x: Zonotope):
        l, u = x.l_u_bound
        zero_tensor = torch.zeros_like(x.center)
        negative_mask = (u < 0)
        crossing_mask = (l < 0) & (u > 0)

        threshold = u / (u - l)
        if self.first_run:
            self.first_run = False
            self.slope_param.data = threshold
        slope = self.slope_param.clamp(0, 1)
        new_generator_0 = (1 - slope) * u * 0.5 * crossing_mask.float()
        new_generator_1 = -slope * l * 0.5 * crossing_mask.float()
        new_generator = torch.where(slope <= threshold, new_generator_0, new_generator_1)

        new_center = torch.where(crossing_mask, x.center * slope + new_generator, x.center)
        new_center = torch.where(negative_mask, zero_tensor, new_center)

        compressed_generator_indices = crossing_mask.nonzero(as_tuple=True)
        num_activations = compressed_generator_indices[0].size(0)
        do_not_repeat_other_dims = [1] * x.center.dim()
        stacked_generator_indices = (torch.arange(num_activations), *compressed_generator_indices)

        new_eps_terms = zero_tensor.unsqueeze(0).repeat(num_activations, *do_not_repeat_other_dims)
        new_eps_terms[stacked_generator_indices] = new_generator[compressed_generator_indices]

        old_new_generators = torch.where(crossing_mask, x.generators * slope, x.generators)
        old_new_generators = torch.where(negative_mask, torch.zeros_like(x.generators), old_new_generators)

        return Zonotope(new_center, torch.cat((old_new_generators, new_eps_terms), dim=0))
