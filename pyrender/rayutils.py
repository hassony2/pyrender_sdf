import torch

def create_ray(f_x, f_y, c_x, c_y, p_x, p_y):
    direction = torch.Tensor([f_y * (p_x  - c_x), f_x * (p_y - c_y), f_x * f_y])
    direction = direction / torch.norm(direction)
    origin = torch.Tensor([0, 0, 0])
    return origin, direction

def shoot_ray(origin, direction, obj, stop_epsilon=0.01, max_dist=100):
    stop = False
    intersect = False
    point = origin
    step_nb = 0
    while not stop:
        obj_dist = obj.eval_sdf(point)
        if torch.norm(point - origin) > max_dist:
            stop = True
        elif obj_dist < 0:
            warnings.warn('Origin inside object !!')
        elif obj_dist < stop_epsilon:
            stop = True
            intersect = True
        else:
            point = point + obj_dist * direction
            step_nb = step_nb + 1
    return intersect, point, step_nb
