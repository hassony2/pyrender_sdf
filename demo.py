import time

from matplotlib import pyplot as plt
import torch

from pyrender import render, testutils

p_xmax, p_ymax = 200, 120
f_x, f_y = 600, 600
c_x, c_y = int(p_xmax / 2), int(p_ymax / 2)
sphere = testutils.Sphere(z_offset=10, radius=0.5)

# Render silhouette
start = time.time()
image = render.render_silhouette(f_x, f_y, c_x, c_y, p_xmax=p_xmax, p_ymax=p_ymax, obj=sphere, max_dist=12)
end = time.time()
print('Rendering silhouette took {:.03f} seconds for image of size {}x{}'.format((end - start) * 1, p_xmax, p_ymax))
plt.imshow(image.permute(1, 0))
plt.show()

# Render normals
start = time.time()
image = render.render_normals(f_x, f_y, c_x, c_y, p_xmax=p_xmax, p_ymax=p_ymax, obj=sphere, max_dist=12)
end = time.time()
print('Rendering normals took {:.03f} seconds for image of size {}x{}'.format((end - start) * 1, p_xmax, p_ymax))
plt.imshow((image.permute(2, 1, 0) + 1) / 2)
plt.show()

