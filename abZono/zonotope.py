import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Zonotope:

    def __init__(self, center, generators):
        self.center = center
        self.generators = generators

    def to_device(self, device):
        self.center = self.center.to(device)
        self.generators = self.generators.to(device)

    def __add__(self, other):
        if isinstance(other, Zonotope):
            return Zonotope(self.center + other.center, self.generators + other.generators)
        else:
            return Zonotope(self.center + other, self.generators)

    def __matmul__(self, other):
        return Zonotope(self.center @ other, self.generators @ other.abs().t())

    def __mul__(self, other):
        return Zonotope(self.center * other, self.generators * other)

    def relu(self):
        #Bei mehreren Durchläufen max min (oder min max) speichern und wiederverwenden
        l = self.center - self.generators.abs().sum()
        u = self.center + self.generators.abs().sum()
        if torch.all(l > 0):
            return self
        elif torch.all(u < 0):
            return Zonotope(torch.zeros_like(self.center), torch.zeros_like(self.generators))
        else:
            # TODO: check if this is correct, sizes of new_error_term should be (1, 1, 1, 1) (same size as center and
            #  generators) or? NO: Add dimension to generators
            slope = u / (u - l)
            new_error_term = -slope * l * 0.5
            return Zonotope(slope * self.center + new_error_term, torch.cat(self.generators, new_error_term))

    def add_alpha(self):
        #MAYBE: sigmoid of unclamped variable
        self.slope = Variable(torch.clamp(self.slope_threshold, 0, 1), requires_grad=True)
        self.slope.retain_grad()

    @property
    def slope_threshold(self) -> torch.Tensor:
        return self.upper_bound / (self.upper_bound - self.lower_bound)

    def size(self):
        return self.center.size()

    def __repr__(self):
        return "Zonotope(center={}, generators={})".format(self.center, self.generators)

    def __str__(self):
        return self.__repr__()