import torch


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

    def get_label(self):
        return torch.argmax(self.center + self.generators.abs().sum(dim=0))

    @property
    def slope_threshold(self) -> torch.Tensor:
        l = self.center - self.generators.abs().sum(dim=0)
        u = self.center + self.generators.abs().sum(dim=0)
        return u / (u - l)

    @property
    def size(self):
        return self.center.size()

    @property
    def device(self):
        return self.center.device

    def __repr__(self):
        return "Zonotope(center={}, generators={})".format(self.center, self.generators)

    def __str__(self):
        return self.__repr__()
