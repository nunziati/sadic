from PDBEntity import PDBEntity
from Multisphere import Multisphere
from Sphere import Sphere

from Quantizer import RegularStepsCartesianQuantizer
import numpy as np
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

protein = PDBEntity("1GWD")
max_distance = 103.
probe_radius = 11.06 # r0
steps_number = 16 # nice value = 32
protein_multisphere = Multisphere(protein)

"""durations = []
errors = []

for size in tqdm(range(16, 100)):

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
ax1.plot(range(16, 100), durations)
ax2 = f.add_subplot(2, 1, 2)
ax2.plot(range(16, 100), errors)

plt.show()"""

def sadic_cubes(steps_number, protein_multisphere):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in tqdm(enumerate(augmented_centers)):
        to_select = (
            (my_centers[:, 0] >= -2. - probe_radius) &
            (my_centers[:, 0] <= 2. + probe_radius) &
            (my_centers[:, 1] >= -2. - probe_radius) &
            (my_centers[:, 1] <= 2. + probe_radius) &
            (my_centers[:,2] >= -2. - probe_radius) &
            (my_centers[:,2] <= 2. + probe_radius)
        )
        selected_centers = my_centers[to_select]
        selected_radii = squared_radii[to_select]

        depth_idx[idx] = 2 / reference_volume * (((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= selected_radii).any(axis=0).sum() * volume

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx

def sadic_norm(steps_number, protein_multisphere):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    radii = radii.reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in tqdm(enumerate(augmented_centers)):
        depth_idx[idx] = 2 / reference_volume * (np.linalg.norm(points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3)), ord=2, axis=2) <= radii.astype(np.float32)).any(axis=0).sum() * volume

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx


def sadic(steps_number, protein_multisphere, probe_radius):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in tqdm(enumerate(augmented_centers)):
        depth_idx[idx] = 2 / reference_volume * (((points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= squared_radii).any(axis=0).sum() * volume

    # max_depth = np.max(depth_idx)
    # count how many elements of depth_idx are equals to max_depth
    count = depth_idx[depth_idx == 2.].shape[0]

    print(count)
    # print(np.sort(depth_idx)[::-1][:10])
    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()
    print("fine sadic")"""

    return depth_idx, count

def sadic_one_shot(steps_number, protein_multisphere):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    radii = radii.reshape((-1, 1)).astype(np.float16)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float16)

    reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    a = (np.linalg.norm(points.reshape((1, -1, 3)) - augmented_centers.reshape((-1, 1, 3)), ord=2, axis=2) <= radii).any(axis=0).sum() * volume

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return a

def test_time(func, *args):
    start_time = time()
    func(*args)
    duration = time() - start_time
    return duration


# test time of the two functions sadic and sadic_cubes
# sadic(16, protein_multisphere)
"""print(test_time(sadic_cubes, 16, protein_multisphere))
print(test_time(sadic_norm, 16, protein_multisphere))
print(test_time(sadic_one_shot, 16, protein_multisphere))"""

a = 10.
b = 20.
step = b - a

while b - a > 0.00001:
    print(f"Trying interval [{a}, {b}].")
    result = sadic(16, protein_multisphere, (a + b) / 2.)
    if result[1] == 0:
        b = (a + b) / 2.
    elif result[1] == 1:
        b = b + (a + b) / 2.
    else:
        a = (a + b) / 2.

print((a + b) / 2.)

"""a = input()

true_depth_index = sadic(16, protein_multisphere, probe_radius)
np.save("true_depth_index.npy", true_depth_index)
step_values = np.arange(5, 20)

duration = []
approximate_depth_index = np.empty((len(step_values), true_depth_index.shape[0]), dtype=np.float32)

for idx, step_value in tqdm(enumerate(step_values)):
    print(step_value)
    start_time = time()
    approximate_depth_index[idx] = sadic(int(step_value), protein_multisphere, probe_radius)
    duration.append(time() - start_time)

mean_erorr = np.mean(np.abs(approximate_depth_index - true_depth_index), axis=1)
rmse = np.sqrt(np.mean(np.square(approximate_depth_index - true_depth_index), axis=1))

f = plt.figure(figsize=(10, 5))


ax1 = f.add_subplot(3, 1, 1)
ax1.plot(step_values, duration)
ax2 = f.add_subplot(3, 1, 2)
ax2.plot(step_values, mean_erorr)
ax3 = f.add_subplot(3, 1, 3)
ax3.plot(step_values, rmse)
plt.show()
"""

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