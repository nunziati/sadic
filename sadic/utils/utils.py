import numpy as np

from sadic.utils.typing import PointType, is_PointType

def point_square_distance(x, y) -> float:
    if not is_PointType(x) or not is_PointType(y):
        raise TypeError("x and y must be PointType")
    
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2

def pairwise_distance(a, b):
    n, m = a.shape[0], b.shape[0]
    a_squared = np.sum(a**2, axis=1)
    b_squared = np.sum(b**2, axis=1)
    ab = np.dot(a, b.T)

    return a_squared[:, np.newaxis] + b_squared[np.newaxis, :] - 2*ab

def pairwise_distance_jit(a, b):
    n, m = a.shape[0], b.shape[0]
    a_squared = np.sum(a**2, axis=1)
    b_squared = np.sum(b**2, axis=1)
    ab = np.dot(a, b.T)
    distance = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            distance[i,j] = np.sqrt(a_squared[i] + b_squared[j] - 2*ab[i,j])

    return distance