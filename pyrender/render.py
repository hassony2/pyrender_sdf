import torch
import numpy as np

from pyrender import rayutils

def render_silhouette(f_x, f_y, c_x, c_y, obj, p_xmax=112, p_ymax=112, max_dist=100):
    image = torch.zeros(p_xmax, p_ymax)
    for p_x in range(p_xmax):
        for p_y in range(p_ymax):
            ray_or, ray_dir = rayutils.create_ray(f_x, f_y, c_x, c_y, p_x, p_y)
            intersect, inter_point, step_nb = rayutils.shoot_ray(ray_or, ray_dir, obj, max_dist=100)
            if intersect:
                image[p_x, p_y] = 1
    return image

def render_normals(f_x, f_y, c_x, c_y, obj, p_xmax=112, p_ymax=112, max_dist=100):
    image = torch.zeros(3, p_xmax, p_ymax)
    for p_x in range(p_xmax):
        for p_y in range(p_ymax):
            ray_or, ray_dir = rayutils.create_ray(f_x, f_y, c_x, c_y, p_x, p_y)
            intersect, inter_point, step_nb = rayutils.shoot_ray(ray_or, ray_dir, obj, max_dist=max_dist)
            if intersect:
                normal = obj.eval_sdf_grad(inter_point)
                image[:, p_x, p_y] = normal
    return image


