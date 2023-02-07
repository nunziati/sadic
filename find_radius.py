from PDBEntity import PDBEntity
from Multisphere import Multisphere
from Sphere import Sphere

from Quantizer import RegularStepsCartesianQuantizer
from Quantizer import RegularStepsSphericalQuantizer

import numpy as np
from time import time
# import matplotlib.pyplot as plt
from tqdm import tqdm

from mytyping import PointType
from mytyping import is_PointType
from utils import *

protein = PDBEntity("1GWD")

probe_radius = 11.06
steps_number = 16 # nice value = 32
rho_steps_number = 2
theta_steps_number = 360
phi_steps_number = 180


protein_multisphere = Multisphere(protein)

quantizer = RegularStepsSphericalQuantizer(rho_steps_number, theta_steps_number, phi_steps_number)
quantizer1 = RegularStepsCartesianQuantizer(steps_number)
#sphere = Sphere(np.array([0., 0., 0.]), probe_radius)
#points, volume = quantizer.get_points_and_volumes(sphere)
centers, radii = protein_multisphere.get_all_centers_and_radii()
#radii = radii.reshape((-1, 1)).astype(np.float32)
#reference_volume = volume * points.shape[0]

#print(points.shape)
#print(centers.shape)

#max_distance = 0
#distance = 0
"""
for i in range(centers.shape[0]):
    for j in range(i+1, centers.shape[0]):
        distance = np.linalg.norm(augmented_centers[i] - augmented_centers[j])
        if distance > max_distance:
            max_distance = distance
"""

"""
for i in range(centers.shape[0]):
    for j in range(i+1, centers.shape[0]):
        distance = point_square_distance(centers[i], centers[j])
        if distance > max_distance:
            max_distance = distance
print(max_distance)

#max_sphere = Sphere(np.array([0., 0., 0.]), max_distance/2 + 2)
"""

min_radius = 1.52
max_radius = min_radius
max_radius_atom = 0

deepest_atoms = [0]

#a = time()
for idx in range(centers.shape[0]):
    radius = min_radius * 2

    sphere = Sphere(centers[idx], radius)
    points = quantizer.get_surface_points(sphere)
    while protein_multisphere.is_inside(points).all():
        radius = radius * 2
        sphere = Sphere(centers[idx], radius)
        points = quantizer.get_surface_points(sphere)

    last_fitting_radius = radius / 2

    if last_fitting_radius > max_radius:
        deepest_atoms.clear()
        max_radius = last_fitting_radius

    if last_fitting_radius == max_radius:
        deepest_atoms.append(idx)

print(deepest_atoms)
print(len(deepest_atoms))

candidate_centers = centers[deepest_atoms]

max_radii = []


for candidate in candidate_centers:
    a = max_radius 
    b = max_radius * 2
    
    while b - a > 1:
        sphere = Sphere(candidate, (a + b) / 2)
        points = quantizer.get_surface_points(sphere)
        if protein_multisphere.is_inside_fast(points).all():
            a = (a + b) / 2
        else:
            b = (a + b) / 2

    max_radii.append((a + b) / 2)
    
print(max_radii)
print(max(max_radii))
        
        

    # print(f"current atom: {idx}, max_radius: {radius / 2}")
    



"""max_radius = 1.52
max_radius_atom = 0

for idx in range(centers.shape[0]):
    radius = max_radius * 2

    sphere = Sphere(centers[idx], radius)
    points, volumes = quantizer1.get_points_and_volumes(sphere)
    while protein_multisphere.is_inside_fast(points).all():
        radius = radius * 2
        sphere = Sphere(centers[idx], radius)
        points, volumes = quantizer1.get_points_and_volumes(sphere)

    
    if radius > max_radius * 2:
        max_radius = radius / 2
        max_radius_atom = idx

    print(f"current atom: {idx}, max_radius: {max_radius}")"""


    

    