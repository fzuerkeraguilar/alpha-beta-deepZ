import torch
import torch.nn.functional as F


class Zonotope:

    def __init__(self, center, generators):
        # Center has the same shape as the original tensor
        self.center = center
        # Generator has one more dimension than the input, the first dimension is the number of generators
        self.generators = generators

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
        return Zonotope(self.center - other.center, self.generators - other.generators)

    def __matmul__(self, other):
        if isinstance(other, Zonotope):
            return Zonotope(self.center @ other.center, self.generators @ other.center + other.generators @ self.center)
        return Zonotope(self.center @ other, self.generators @ other)

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
        gen_dim0 = dim0 + 1  # if dim0 != 0 else dim0
        gen_dim1 = dim1 + 1  # if dim1 != 0 else dim1
        transposed_generators = self.generators.transpose(gen_dim0, gen_dim1)

        return Zonotope(transposed_center, transposed_generators)

    def flatten(self, start_dim=0, end_dim=-1):
        # Flatten the center
        flattened_center = self.center.flatten(start_dim, end_dim)

        # Adjust the dimensions for the generators and then flatten
        # The first dimension (number of generators) is not included in the flattening
        gen_start_dim = start_dim + 1 if start_dim != 0 else start_dim  # TODO: check if this is correct
        gen_end_dim = end_dim + 1 if end_dim != -1 else end_dim  # TODO: check if this is correct
        flattened_generators = self.generators.flatten(
            gen_start_dim, gen_end_dim)

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

    def normalize(self, dim, p=2):
        return Zonotope(self.center / self.center.norm(p, dim=dim, keepdim=True),
                        self.generators / self.center.norm(p, dim=dim, keepdim=True))

    def l_inf_norm(self):
        return self.generators.abs().sum(dim=0)

    def l_inf_loss(self, target):
        return (self.l_inf_norm() - target.l_inf_norm()).abs().sum()

    def min_diff(self, true_label):
        min_value_of_true_label = torch.full_like(self.center, (
                self.center[true_label] - self.generators.abs().sum(dim=0)[true_label]).item())
        return min_value_of_true_label - (self.center + self.generators.abs().sum(dim=0))

    def label_loss(self, target_label):
        return torch.clamp(self.min_diff(target_label), min=0).sum()

    # spec, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
    def vnnlib_loss(self, spec):
        loss = torch.tensor(0.0, device=self.device)
        for factor, rhs in spec:
            positive_factor = torch.clamp(factor, min=0)
            negative_factor = torch.clamp(factor, max=0)
            loss += torch.sum(positive_factor * self.upper_bound - rhs)
            loss += torch.sum(negative_factor * self.lower_bound - rhs)
        return loss

    def contains(self, other: 'Zonotope'):
        return (self.lower_bound <= other.lower_bound).all() and (self.upper_bound >= other.upper_bound).all()

    def contains_point(self, point):
        return (self.lower_bound <= point).all() and (self.upper_bound >= point).all()

    def random_point(self):
        return torch.rand_like(self.center) * (self.upper_bound - self.lower_bound) + self.lower_bound

    def to_device(self, device):
        self.center = self.center.to(device)
        self.generators = self.generators.to(device)

    def to(self, *args, **kwargs):
        return Zonotope(self.center.to(*args, **kwargs), self.generators.to(*args, **kwargs))

    def size(self):
        return self.center.size()

    def get_label(self):
        return torch.argmax(self.center + self.generators.abs().sum(dim=0))

    @property
    def slope_threshold(self) -> torch.Tensor:
        l = self.center - self.generators.abs().sum(dim=0)
        u = self.center + self.generators.abs().sum(dim=0)
        return u / (u - l)

    @property
    def lower_bound(self):
        return self.center - self.generators.abs().sum(dim=0)

    @property
    def upper_bound(self):
        return self.center + self.generators.abs().sum(dim=0)

    @property
    def shape(self):
        return self.center.shape

    @property
    def device(self):
        return self.center.device

    @staticmethod
    def from_vnnlib(l_u_list, shape, dtype):
        centers = []
        generators = []
        for l, u in l_u_list:
            centers.append((l + u) / 2)
            generators.append((u - l) / 2)

        center = torch.tensor(centers, dtype=dtype).reshape(shape)
        generators = torch.tensor(generators, dtype=dtype).reshape(1, *shape)
        center = center.clone().detach().requires_grad_(True)
        generators = generators.clone().detach().requires_grad_(True)
        return Zonotope(center, generators)

    @staticmethod
    def from_l_inf(center, radius):
        return Zonotope(center, (torch.ones_like(center) * radius).unsqueeze(0))

    @staticmethod
    def zeros_like(zonotope):
        return Zonotope(torch.zeros_like(zonotope.center), torch.zeros_like(zonotope.generators).unsqueeze(0))

    @staticmethod
    def ones_like(zonotope):
        return Zonotope(torch.ones_like(zonotope.center), torch.zeros_like(zonotope.generators).unsqueeze(0))

    def __repr__(self):
        return "Zonotope(center={}, generators={})".format(self.center, self.generators)

    def __str__(self):
        return self.__repr__()
