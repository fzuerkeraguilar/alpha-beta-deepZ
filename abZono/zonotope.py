from email import generator
from locale import normalize
from math import perm
import re
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
    
    def reshape(self, shape):
        return Zonotope(self.center.reshape(shape), self.generators.reshape(shape))
    
    def permute(self, *dims):
        return Zonotope(self.center.permute(*dims), self.generators.permute(*dims))
    
    def normalize(self, dim, p=2):
        return Zonotope(self.center / self.center.norm(p, dim=dim, keepdim=True), self.generators / self.center.norm(p, dim=dim, keepdim=True))
    
    

    def l_inf_norm(self):
        return self.generators.abs().sum(dim=0)
    
    def l_inf_loss(self, target):
        return (self.l_inf_norm() - target.l_inf_norm()).abs().sum()
    
    def min_diff(self, true_label):
        min_value_of_true_label = torch.full_like(self.center, (self.center[true_label] - self.generators.abs().sum(dim=0)[true_label]).item())
        print(min_value_of_true_label)
        return min_value_of_true_label - (self.center + self.generators.abs().sum(dim=0))
    
    def label_loss(self, target_label):
        return torch.clamp(self.min_diff(target_label), min=0).sum()

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

    @staticmethod
    def from_l_inf(center, radius):
        return Zonotope(center, torch.ones_like(center) * radius)
    
    @staticmethod
    def zeros_like(zonotope):
        return Zonotope(torch.zeros_like(zonotope.center), torch.zeros_like(zonotope.generators))
    
    @staticmethod
    def ones_like(zonotope):
        return Zonotope(torch.ones_like(zonotope.center), torch.zeros_like(zonotope.generators))

    def __repr__(self):
        return "Zonotope(center={}, generators={})".format(self.center, self.generators)

    def __str__(self):
        return self.__repr__()
