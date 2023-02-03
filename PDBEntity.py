from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb
import numpy as np

class PDBEntity:
    def __init__(self, id):
        self.entity = pd().fetch_pdb(id)
        self.atom_types = self.entity.df["ATOM"]["element_symbol"]
        self.atoms = self.entity.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy()

    def get_centers(self):
        return self.atoms

    def get_radii(self):
        vdw_radii = {
            "C": 1.7,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "P": 1.8,
            "S": 1.8,
        }

        probe_radius = 1.52

        return np.array([vdw_radii[atom_type] + probe_radius for atom_type in self.atom_types])