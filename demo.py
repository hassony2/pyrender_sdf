from matplotlib import pyplot as plt
import torch

from pyrender import render, testutils

p_xmax, p_ymax = 200, 120
f_x, f_y = 600, 600
c_x, c_y = int(p_xmax / 2), int(p_ymax / 2)
sphere = testutils.Sphere(z_offset=10, radius=0.5)
image = render.render_normals(f_x, f_y, c_x, c_y, p_xmax=p_xmax, p_ymax=p_ymax, obj=sphere, max_dist=12)
plt.imshow((image.permute(2, 1, 0) + 1) / 2)
plt.show()
