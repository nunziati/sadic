from Multisphere import Multisphere

def find_reference_radius(multisphere, phi_values, theta_values):
    pass

def find_depth_index(multisphere, point, reference_radius):
    pass

def sadic(protein):
    multisphere = Multisphere(protein)
    reference_radius = find_reference_radius(multisphere)
    
    depth_index_list = []
    for point in multisphere:
        depth_index_list.append(find_depth_index(multisphere, point, reference_radius))
