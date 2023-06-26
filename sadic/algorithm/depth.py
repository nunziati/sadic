from sadic.solid import Sphere, VoxelSolid
from sadic.quantizer import RegularStepsSphericalQuantizer, RegularStepsCartesianQuantizer
import numpy as np

default_quantizer_class = RegularStepsSphericalQuantizer
default_quantizer_kwargs = {"rho_steps_number": 10, "theta_steps_number": 36, "phi_steps_number": 18}

def sadic_cubes(protein_multisphere, probe_radius, steps_number):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    # reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in enumerate(augmented_centers):
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

        depth_idx[idx] = 2 * (((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= selected_radii).any(axis=0).sum() / points.shape[0]

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx, -1

def sadic_cubes_optimized(protein_multisphere, probe_radius, steps_number):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    # reference_volume = volume * points.shape[0]

    to_select = (
        (augmented_centers[:, :, 0] >= -2. - probe_radius) &
        (augmented_centers[:, :, 0] <= 2. + probe_radius) &
        (augmented_centers[:, :, 1] >= -2. - probe_radius) &
        (augmented_centers[:, :, 1] <= 2. + probe_radius) &
        (augmented_centers[:, :, 2] >= -2. - probe_radius) &
        (augmented_centers[:, :, 2] <= 2. + probe_radius)
    )

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in enumerate(augmented_centers):
        selected_centers = my_centers[to_select[idx]]
        selected_radii = squared_radii[to_select[idx]]

        depth_idx[idx] = 2 * (((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= selected_radii).any(axis=0).sum() / points.shape[0]

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx, -1

def sadic_sphere(protein_multisphere, probe_radius, steps_number):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    # reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in enumerate(augmented_centers):
        to_select = (my_centers ** 2).sum(axis=-1) <= (2. + probe_radius) ** 2
        selected_centers = my_centers[to_select]
        selected_radii = squared_radii[to_select]

        depth_idx[idx] = 2 * (((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= selected_radii).any(axis=0).sum() / points.shape[0]

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx, -1

def sadic_sphere_optimized(protein_multisphere, probe_radius, steps_number):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    # reference_volume = volume * points.shape[0]

    to_select = (augmented_centers ** 2).sum(axis=-1) <= (3.5 + probe_radius) ** 2

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in enumerate(augmented_centers):
        selected_centers = my_centers[to_select[idx]]
        selected_radii = squared_radii[to_select[idx]]

        depth_idx[idx] = 2 * (((points.reshape((1, -1, 3)) - selected_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= selected_radii).any(axis=0).sum() / points.shape[0]

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx, -1

def sadic_norm(protein_multisphere, probe_radius, steps_number):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    radii = radii.reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)
    reference_volume = volume * points.shape[0]

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in enumerate(augmented_centers):
        depth_idx[idx] = 2 / reference_volume * (np.linalg.norm(points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3)), ord=2, axis=2) <= radii.astype(np.float32)).any(axis=0).sum() * volume

    """print(max(depth_idx))
    plt.hist(depth_idx, bins=30)
    plt.show()"""

    return depth_idx

def sadic_original(protein_multisphere, probe_radius, steps_number):
    quantizer = RegularStepsCartesianQuantizer(steps_number)

    sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
    points, volume = quantizer.get_points_and_volumes(sphere)
    centers, radii = protein_multisphere.get_all_centers_and_radii()
    squared_radii = (radii ** 2).reshape((-1, 1)).astype(np.float32)
    augmented_centers = (centers.reshape(1, -1, 3) - centers.reshape(-1, 1, 3)).astype(np.float32)

    depth_idx = np.empty(centers.shape[0], dtype=np.float32)

    for idx, my_centers in enumerate(augmented_centers):
        depth_idx[idx] = 2  * (((points.reshape((1, -1, 3)) - my_centers.reshape((-1, 1, 3))) ** 2).sum(axis=-1) <= squared_radii).any(axis=0).sum() / points.shape[0]

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

def sadic_one_shot(protein_multisphere, probe_radius, steps_number):
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


def sadic_original_voxel(protein_solid: VoxelSolid, probe_radius: float):
    """from tqdm import tqdm
    centers = protein_solid.multisphere.get_all_centers()

    center_number = centers.shape[0]
    depth_idx = np.empty(center_number, dtype=np.float32)

    for idx, center in tqdm(enumerate(centers)):
        sphere = VoxelSolid([Sphere(center, probe_radius)], resolution=protein_solid.resolution, extreme_coordinates=protein_solid.extreme_coordinates)
        sphere_volume = sphere.int_volume()
        sphere.intersection_(protein_solid)
        depth_idx[idx] = 2 * (1 - sphere.int_volume() / sphere_volume)

    count = depth_idx[depth_idx == 0.].shape[0]"""

    """from tqdm import tqdm
    centers = protein_solid.multisphere.get_all_centers()

    center_number = centers.shape[0]
    origin = protein_solid.grid_to_cartesian(protein_solid.dimensions // 2)

    origin_sphere = VoxelSolid([Sphere(origin, probe_radius)], resolution=protein_solid.resolution, extreme_coordinates=protein_solid.extreme_coordinates)
    sphere_volume = origin_sphere.int_volume()

    depth_idx = np.empty(center_number, dtype=np.float32)

    for idx, center in tqdm(enumerate(centers)):
        sphere = origin_sphere.translate(protein_solid.cartesian_to_grid(center) - protein_solid.cartesian_to_grid(origin))
        sphere.intersection_(protein_solid)
        depth_idx[idx] = 2 * (1 - sphere.int_volume() / sphere_volume)

    count = depth_idx[depth_idx == 0.].shape[0]"""

    from tqdm import tqdm
    centers = protein_solid.multisphere.get_all_centers()

    center_number = centers.shape[0]

    depth_idx = np.empty(center_number, dtype=np.float32)

    for idx, center in tqdm(enumerate(centers)):
        sphere = VoxelSolid([Sphere(center, probe_radius)], resolution=protein_solid.resolution, align_with=protein_solid)
        sphere_volume = sphere.int_volume()
        sphere.intersection_(protein_solid)
        intersection_volume = sphere.int_volume()
        depth_idx[idx] = 2 * (1 - intersection_volume / sphere_volume)

    count = depth_idx[depth_idx == 0.].shape[0]

    print(count)
    print("min depth index = ", np.min(depth_idx))
    return depth_idx, count