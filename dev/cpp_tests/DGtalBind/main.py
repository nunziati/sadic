import numpy as np
import sys

sys.path.append("/home/giacomo/sadic/dev/cpp_tests/DGtalBind/build")
import my_module

# Create a numpy array
shape = (11, 11, 11)  # Example dimensions
np_array = np.zeros(shape, dtype=np.int32)

center = np.array([5, 5, 5], dtype=np.int32)

# fill a sphere of radius 4
for i in range(11):
    for j in range(11):
        for k in range(11):
            if np.linalg.norm(np.array([i, j, k], dtype=np.int32) - center) <= 4:
                np_array[i, j, k] = 1

print(np_array.shape)
print(np_array.dtype)


# Define a list of points
points = [(4, 3, 2)]  # Example points

# Compute the maximum distance
max_distance = my_module.compute_max_distance(np_array, points)
print("Maximum distance:", max_distance)