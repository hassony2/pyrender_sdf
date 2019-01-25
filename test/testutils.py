import numpy as np

def sphere_sdf(x, y, z, z_offset=200, radius=1):
    sdf = np.sqrt(x**2 + y**2 + (z - z_offset)**2) - radius ** 2
    return sdf
    
