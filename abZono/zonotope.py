import torch
import torch.nn.functional as F


class Zonotope:

    def __init__(self, center, generators, optimize_slope = False):
        self.center = center
        self.generators = generators
        self.optimize_slope = optimize_slope
        self.center.requires_grad = False
        if self.optimize_slope:
            self.slope = torch.zeros_like(center, requires_grad=True)

    def __add__(self, other):
        return Zonotope(self.center + other.center, self.generators + other.generators)

    def __sub__(self, other):
        return Zonotope(self.center - other.center, self.generators - other.generators)

    def __mul__(self, other):
        return Zonotope(self.center * other, self.generators * other)

    def to_device(self, device):
        self.center = self.center.to(device)
        self.generators = self.generators.to(device)

    @property
    def slope_threshold(self) -> torch.Tensor:
        return self.upper_bound / (self.upper_bound - self.lower_bound)

    def size(self):
        return self.center.size()

    def __repr__(self):
        return "Zonotope(center={}, generators={})".format(self.center, self.generators)

    def __str__(self):
        return self.__repr__()
