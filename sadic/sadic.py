from sadic.solid import Multisphere
from sadic.pdb import PDBEntity
from sadic.algorithm.radius import find_max_radius_point
from sadic.algorithm.depth import sadic_sphere
from sadic.utils.config import default_steps_number as steps_number

def sadic(input):
    print("Loading protein".ljust(30, "."), end="", flush=True)
    protein = PDBEntity(input)
    print("DONE")
    print("Creating multisphere".ljust(30, "."), end="", flush=True)
    protein_multisphere = Multisphere(protein)
    print("DONE")
    print("Computing max radius".ljust(30, "."), end="", flush=True)
    _, probe_radius = find_max_radius_point(protein_multisphere, bisection_threshold = 0.2)
    print("DONE")
    print("Computing depth indexes".ljust(30, "."), end="", flush=True)
    true_depth_index = sadic_sphere(protein_multisphere, probe_radius, steps_number)[0]
    print("DONE")
    return true_depth_index