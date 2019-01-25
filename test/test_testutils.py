import numpy as np
import torch

from pyrender import testutils

def test_sphere_sdf():
    sphere = testutils.Sphere(z_offset=0, radius=2)
    point = torch.Tensor([1, 0, 0])
    sdf = sphere.eval_sdf(point)
    assert sdf == -1

def test_sphere_sdf():
    sphere = testutils.Sphere(z_offset=2, radius=0.5)
    point = torch.Tensor([0, 0, 0])
    sdf = sphere.eval_sdf(point)
    assert sdf == 1.5

def test_sphere_sdf_grad():
    sphere = testutils.Sphere(z_offset=0, radius=2)
    point = torch.Tensor([1, 0, 0])
    sdf_grad = sphere.eval_sdf_grad(point)
    assert torch.all(torch.eq(sdf_grad, torch.Tensor([1, 0, 0])))

def test_sphere_sdf_grad_norm():
    sphere = testutils.Sphere(z_offset=0, radius=2)
    point = torch.Tensor([1, 2, 4])
    sdf_grad = sphere.eval_sdf_grad(point)
    assert np.isclose(torch.norm(sdf_grad).item(), 1)

def test_sphere_sdf_grad_norm_withoffset():
    sphere = testutils.Sphere(z_offset=2.5, radius=2)
    point = torch.Tensor([1, 2, 4])
    sdf_grad = sphere.eval_sdf_grad(point)
    assert np.isclose(torch.norm(sdf_grad).item(), 1)
