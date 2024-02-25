import torch.nn
from abZono import Zonotope


class ZonoBatchNorm2d(torch.nn.Module):
    def __init__(self, layer: torch.nn.BatchNorm2d):
        super().__init__()
        self.num_features = layer.num_features
        self.eps = layer.eps
        self.momentum = layer.momentum
        self.affine = False
        self.track_running_stats = False
        self.__name__ = self.__class__.__name__

        self.register_buffer("running_mean", layer.running_mean)
        self.register_buffer("running_var", layer.running_var)
        self.register_buffer("weight", layer.weight.data)
        self.register_buffer("bias", layer.bias.data)

    def forward(self, x: Zonotope):
        if self.training:
            mean = x.center.mean(dim=[0, 2, 3], keepdim=True)
            var = (x.center - mean).pow(2).mean(dim=[0, 2, 3], keepdim=True)
            x.center = (x.center - mean) / (var + self.eps).sqrt()
            x.generators = x.generators / (var + self.eps).sqrt()
        else:
            x.center = (x.center - self.running_mean) / (self.running_var + self.eps).sqrt()
            x.generators = x.generators / (self.running_var + self.eps).sqrt()

        x.center = x.center * self.weight + self.bias
        x.generators = x.generators * self.weight

        return x
