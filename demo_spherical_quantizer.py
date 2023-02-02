from PDBEntity import PDBEntity
from Multisphere import Multisphere
from Sphere import Sphere
from Quantizer import RegularStepsCartesianQuantizer
import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

protein = PDBEntity("3BJ1")
max_distance = 103.
probe_radius = 2.
steps_number = 10 # nice value
protein_multisphere = Multisphere(protein)

"""durations = []
errors = []

for size in tqdm(range(20, 100)):

    quantizer = RegularStepsCartesianQuantizer(size)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    true_volume = probe_radius ** 3 * np.pi * 4. / 3.

    start_time = time()
    points, volume = quantizer.get_points_and_volumes(sphere)
    duration = time() - start_time

    durations.append(duration)

    approx_volume = points.shape[0] * volume
    errors.append(abs(approx_volume - true_volume) / true_volume * 100)

f = plt.figure()

ax1 = f.add_subplot(2, 1, 1)
ax1.plot(range(20, 100), durations)
ax2 = f.add_subplot(2, 1, 2)
ax2.plot(range(20, 100), errors)

plt.show()"""

quantizer = RegularStepsCartesianQuantizer(steps_number)

sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
points, volume = quantizer.get_points_and_volumes(sphere)
centers, radii = protein_multisphere.get_all_centers_and_radii()
squared_radii = (radii ** 2).reshape((-1, 1))
augmented_centers = centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)
reference_volume = volume * points.shape[0]

depth_idx = np.empty(centers.shape[0], dtype=np.float32)

for idx, my_centers in tqdm(enumerate(augmented_centers)):
    depth_idx[idx] = 2 / reference_volume * (((points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= squared_radii).any(axis=0).sum() * volume

print(depth_idx)
print(max(depth_idx))
plt.hist(depth_idx, bins=30)
plt.show()

"""points_list = []

start_time = time()
for center in tqdm(protein_multisphere.get_all_centers_and_radii()[0]):
    points_list.append(quantizer.get_points_and_volumes(Sphere(center, probe_radius))[0])

int_time = time()
print(f"Got all points after {int_time - start_time} seconds")

for points in tqdm(points_list):
    protein_multisphere.is_inside_fast(points)
print(f"is_inside_fast over after {time() - int_time} seconds")
print(f"Total time: {time() - start_time} seconds")"""

"""sadic = Sadic()

d_index = sadic.sadic(protein)"""