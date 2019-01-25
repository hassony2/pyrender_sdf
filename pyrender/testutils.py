import numpy as np
import torch

class Sphere():
    def __init__(self, z_offset=200, radius=1):
        self.radius = radius
        self.offset = torch.Tensor([0, 0, z_offset])

    def eval_sdf(self, xyz):
        sdf = torch.sqrt(((xyz - self.offset) ** 2).sum()) - self.radius
        return sdf

    def eval_sdf_grad(self, xyz):
        denum = torch.sqrt(((xyz - self.offset) ** 2).sum())
        sdf_grad = (xyz - self.offset) / denum
        return sdf_grad
