from sklearn.neighbors.kde import KernelDensity
import numpy as np
bandwidth = 4
bandwidth_inv = 1.0 / bandwidth
x = np.array([3, 2])
c = np.array([1.5, 1.5])
coor_diff = x - c
print coor_diff ** 2
kde_coef = 1.0 / 2 / np.pi / bandwidth
print kde_coef * np.exp(-0.5 * np.sum(coor_diff ** 2) * bandwidth_inv)