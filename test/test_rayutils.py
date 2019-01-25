from matplotlib import pyplot as plt
import numpy as np
import torch

from pyrender import rayutils, render, testutils


def test_create_ray():
    p_x, p_y = 0, 0
    f_x, f_y = 400, 600
    c_x, c_y = 120, 150
    ray_or, ray_dir = rayutils.create_ray(f_x, f_y, c_x, c_y, p_x, p_y)
    assert torch.all(torch.eq(ray_or, torch.Tensor([0, 0, 0])))
    assert torch.norm(ray_dir) == 1



def test_shoot_ray():
    ray_or = torch.Tensor([0, 0, 0])
    ray_dir = torch.Tensor([0, 0, 1])
    sphere = testutils.Sphere(z_offset=2, radius=0.5)
    # Sphere in front of ray
    intersect, inter_point, step_nb = rayutils.shoot_ray(ray_or, ray_dir, sphere)
    assert intersect
    assert torch.all(torch.eq(inter_point, torch.Tensor([0, 0, 1.5])))

    # Sphere in behind ray
    sphere = testutils.Sphere(z_offset=-2, radius=0.5)
    max_dist = 20
    intersect, inter_point, step_nb = rayutils.shoot_ray(ray_or, ray_dir, sphere, max_dist=max_dist)
    assert not intersect
    assert torch.norm(inter_point - ray_or) > max_dist 
