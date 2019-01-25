from matplotlib import pyplot as plt
import torch

from pyrender import render, testutils

def test_render_silhouette():
    p_xmax, p_ymax = 112, 120
    f_x, f_y = 400, 400
    c_x, c_y = int(p_xmax / 2), int(p_ymax / 2)
    sphere = testutils.Sphere(z_offset=10, radius=0.5)
    image = render.render_silhouette(f_x, f_y, c_x, c_y, p_xmax=p_xmax, p_ymax=p_ymax, obj=sphere)
    assert image.shape == (p_xmax, p_ymax)
    assert image[c_x, c_y] == 1
    assert image[0, 0] == 0
    # plt.imshow(image)
    # plt.show()

def test_render_normal():
    p_xmax, p_ymax = 200, 120
    f_x, f_y = 600, 600
    c_x, c_y = int(p_xmax / 2), int(p_ymax / 2)
    sphere = testutils.Sphere(z_offset=10, radius=0.5)
    image = render.render_normals(f_x, f_y, c_x, c_y, p_xmax=p_xmax, p_ymax=p_ymax, obj=sphere, max_dist=12)
    assert image.shape == (3, p_xmax, p_ymax)
    # plt.imshow((image.permute(2, 1, 0) + 1) / 2)
    # plt.show()
    assert torch.all(torch.eq(image[:, c_x, c_y], torch.Tensor([0, 0, -1])))
