import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Zonotope:

    def __init__(self, center, generators):
        self.center = center
        self.generators = generators

    def __add__(self, other):
        return Zonotope(self.center + other.center, self.generators + other.generators)

    def __sub__(self, other):
        return Zonotope(self.center - other.center, self.generators - other.generators)

    def __mul__(self, other):
        return Zonotope(self.center * other, self.generators * other)

    def to_device(self, device):
        self.center = self.center.to(device)
        self.generators = self.generators.to(device)

    def linear(self, weight: torch.Tensor, bias: torch.Tensor):
        return Zonotope(F.linear(self.center, weight, bias), F.linear(self.generators, weight))

    def relu(self):
        # Bei mehreren Durchläufen max min (oder min max) speichern und wiederverwenden
        l = self.center - self.generators.abs().sum(dim=0)
        u = self.center + self.generators.abs().sum(dim=0)
        if torch.all(l > 0):
            return self
        elif torch.all(u < 0):
            return Zonotope(torch.zeros_like(self.center), torch.zeros_like(self.generators))
        else:
            where_crossing = torch.bitwise_and(l < 0, u > 0)
            slope = u / (u - l) * where_crossing
            new_generators = -slope * l * 0.5
            return Zonotope(torch.where(where_crossing, self.center * slope + new_generators, self.center),
                            torch.cat((torch.where(where_crossing, self.generators * slope, self.generators),
                                       new_generators.unsqueeze(0))))

    def conv2d(self, weight: torch.Tensor, bias: torch.Tensor, stride, padding, dilation, groups):
        return Zonotope(F.conv2d(self.center, weight, bias, stride, padding, dilation, groups),
                        F.conv2d(self.generators, weight, None, stride, padding, dilation, groups))

    def add_alpha(self):
        # MAYBE: sigmoid of unclamped variable
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
