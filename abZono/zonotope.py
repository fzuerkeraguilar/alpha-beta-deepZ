import torch
import torch.nn.functional as F
from typing import Tuple
import scipy.optimize as opt
import numpy as np


class Zonotope:

    def __init__(self, center: torch.Tensor, generators: torch.Tensor):
        # Center has the same shape as the original tensor
        self.center: torch.Tensor = center
        # Generator has one more dimension than the input, the first dimension is the number of generators
        self.generators: torch.Tensor = generators

    def __add__(self, other):
        new_center = self.center + other.center
        # If the number of generators is the same, we can simply add them
        if self.generators.shape[0] == other.generators.shape[0]:
            new_generators = self.generators + other.generators
        # Otherwise, we need to pad the one with fewer generators
        elif self.generators.shape[0] < other.generators.shape[0]:
            pad = [0, 0] * (len(self.generators.shape) - 1) + [0, other.generators.shape[0] - self.generators.shape[0]]
            padded_generators = F.pad(self.generators, pad)
            new_generators = padded_generators + other.generators
        else:
            pad = [0, 0] * (len(other.generators.shape) - 1) + [0, self.generators.shape[0] - other.generators.shape[0]]
            padded_generators = F.pad(other.generators, pad)
            new_generators = self.generators + padded_generators

        return Zonotope(new_center, new_generators)

    def __sub__(self, other):
        return Zonotope(self.center - other, self.generators)

    def __div__(self, other):
        return Zonotope(self.center / other, self.generators / other)

    def split(self, j: int):
        jth_generator = self.generators[j]

        # Split the center
        lower_center = self.center - (0.5 * jth_generator)
        upper_center = self.center + (0.5 * jth_generator)

        new_generators = torch.cat([self.generators[:j], self.generators[j] * 0.5, self.generators[j + 1:]], dim=0)

        return Zonotope(lower_center, new_generators), Zonotope(upper_center, new_generators)

    def reshape(self, *shape):
        # Reshape the center
        reshaped_center = self.center.reshape(*shape)

        # Reshape each generator
        generator_shape = [self.generators.shape[0]] + list(shape)
        reshaped_generators = self.generators.reshape(generator_shape)

        return Zonotope(reshaped_center, reshaped_generators)

    def permute(self, *dims):
        # Permute the center
        permuted_center = self.center.permute(*dims)

        # Permute the generators. The first dimension (number of generators) remains unchanged.
        generator_dims = [0] + [dim + 1 for dim in dims]
        permuted_generators = self.generators.permute(*generator_dims)

        return Zonotope(permuted_center, permuted_generators)

    def transpose(self, dim0, dim1):
        # Transpose the center
        transposed_center = self.center.transpose(dim0, dim1)

        # Adjust the dimensions for the generators and then transpose
        gen_dim0 = dim0 + 1 if dim0 < dim1 else dim0
        gen_dim1 = dim1 + 1 if dim1 < dim0 else dim1
        transposed_generators = self.generators.transpose(gen_dim0, gen_dim1)

        return Zonotope(transposed_center, transposed_generators)

    def flatten(self, start_dim=0, end_dim=-1):
        # Flatten the center
        flattened_center = self.center.flatten(start_dim, end_dim)

        # Adjust the dimensions for the generators and then flatten
        # The first dimension (number of generators) is not included in the flattening
        gen_start_dim = start_dim + 1
        flattened_generators = self.generators.flatten(gen_start_dim, end_dim)

        return Zonotope(flattened_center, flattened_generators)

    def pad(self, pad, mode='constant', value=0):
        padded_center = F.pad(self.center, pad, mode, value)

        # Adjust the dimensions for the generators and then pad
        # The first dimension (number of generators) is not included in the padding
        gen_pad = pad + [0, 0]
        padded_generators = F.pad(self.generators, gen_pad, mode, value)

        return Zonotope(padded_center, padded_generators)

    def view(self, *shape):
        # View the center
        viewed_center = self.center.view(*shape)

        # View each generator
        generator_shape = [self.generators.shape[0]] + list(shape)
        viewed_generators = self.generators.view(generator_shape)

        return Zonotope(viewed_center, viewed_generators)

    # spec, provided as a pair (mat, rhs), as in: mat * y <= rhs, where y is the output.
    def vnnlib_loss(self, spec):
        factors, rhs_values, disjunction = spec
        positive_factors = torch.clamp(factors, min=0)
        negative_factors = torch.clamp(factors, max=0)

        # Retrieving l and u bounds
        l, u = self.l_u_bound
        lhs = positive_factors * u + negative_factors * l
        loss = lhs.sum(dim=-1) - rhs_values
        if disjunction:
            return torch.min(loss)  # Because disjunction
        return torch.sum(loss)

    def label_loss(self, label):
        l, u = self.l_u_bound
        mask = torch.ones_like(l)
        mask[:, label] = 0
        non_target = u[mask == 1]

        diff = non_target - l[:, label]
        return F.relu(diff).sum()

    def contains_point(self, point: torch.Tensor):
        point_prime_flat = (point - self.center).flatten()
        generators_flat = self.generators.flatten(start_dim=1).detach().numpy()

        c = np.zeros(generators_flat.shape[0])
        A_eq = generators_flat.T
        b_eq = point_prime_flat.detach().numpy()

        # Solve the linear programming problem
        try:
            result = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(-1, 1))
        except ValueError:
            return False
        return result.success

    def contains_point_box(self, point: torch.Tensor):
        return torch.all(point >= self.lower_bound) and torch.all(point <= self.upper_bound)

    def random_point(self):
        coeffs = 2.0 * torch.rand(len(self.generators)) - 1.0  # Scale to [-1, 1]

        random_point = self.center.clone()
        for i, generator in enumerate(self.generators):
            random_point += coeffs[i] * generator

        return random_point

    def to(self, *args, **kwargs):
        self.center = self.center.to(*args, **kwargs)
        self.generators = self.generators.to(*args, **kwargs)
        return self

    def size(self) -> torch.Size:
        return self.center.size()

    @property
    def num_generators(self):
        return self.generators.size(0)

    @property
    def without_zero_generators(self):
        non_zero_generators = torch.cat([x.unsqueeze(0) for x in self.generators if x.abs().sum() > 0], dim=0)
        return Zonotope(self.center, non_zero_generators)

    @property
    def lower_bound(self) -> torch.Tensor:
        return self.center - self.generators.abs().sum(dim=0)

    @property
    def upper_bound(self) -> torch.Tensor:
        return self.center + self.generators.abs().sum(dim=0)

    @property
    def l_u_bound(self) -> Tuple[torch.Tensor, torch.Tensor]:
        gen_sum = self.generators.abs().sum(dim=0)
        return self.center - gen_sum, self.center + gen_sum

    @property
    def predicted_label(self):
        return torch.argmax(self.lower_bound)

    @property
    def dtype(self) -> torch.dtype:
        return self.center.dtype

    @property
    def shape(self) -> torch.Size:
        return self.center.shape

    @property
    def device(self) -> torch.device:
        return self.center.device

    @staticmethod
    def from_vnnlib(l_u_list, shape: torch.Size, dtype: torch.dtype):
        lower_limits = torch.tensor([x[0] for x in l_u_list], dtype=dtype)
        upper_limits = torch.tensor([x[1] for x in l_u_list], dtype=dtype)

        center = (upper_limits + lower_limits) / 2
        center = center.reshape(shape)

        generators = []

        for i in range(len(lower_limits)):
            generator = torch.zeros_like(center, dtype=dtype)
            generator.view(-1)[i] = (upper_limits[i] - lower_limits[i]) / 2
            generators.append(generator.reshape(shape))

        return Zonotope(center, torch.stack(generators))

    @staticmethod
    def from_l_inf(center: torch.Tensor, epsilon: float, shape: torch.Size, l: float = None, u: float = None):
        if shape:
            center = center.reshape(shape)
        numel = center.numel()
        generators = torch.eye(numel) * epsilon
        generators = generators.reshape(numel, *center.shape)
        if l is None and u is None:
            return Zonotope(center, generators)
        else:
            temp = Zonotope(center, generators)
            temp.to('cpu')
            l_zono, u_zono = temp.l_u_bound
            l_zono = l_zono.clamp(l, u)
            u_zono = u_zono.clamp(l, u)
            return Zonotope((l_zono + u_zono) / 2, (u_zono - l_zono) / 2)


    @staticmethod
    def zeros_like(other: 'Zonotope'):
        return Zonotope(torch.zeros_like(other.center), torch.zeros_like(other.generators).unsqueeze(0))

    @staticmethod
    def ones_like(other: 'Zonotope'):
        return Zonotope(torch.ones_like(other.center), torch.zeros_like(other.generators).unsqueeze(0))

    def __repr__(self):
        return "Zonotope(center={}, generators={})".format(self.center, self.generators)

    def __str__(self):
        return self.__repr__()
