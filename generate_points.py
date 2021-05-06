import sys
import numpy as np
from matplotlib.pyplot import scatter,show


def random_disk(N, r):
    rho = np.sqrt(np.random.uniform(0, r, int(N)))
    phi = np.random.uniform(0, 2 * np.pi, int(N))

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return np.asarray([x, y]).astype(float)

p = random_disk(64 * 1e6, 4.0)

print((p.shape))

scatter(p[0, :], p[1, :], s=0.1)

show()
