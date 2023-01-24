from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb
import numpy as np

class PDBEntity:
    def __init__(self, id):
        self.entity = pd().fetch_pdb(id)
        print(self.entity.df["ATOM"])
        atom_types = self.entity.df["ATOM"]["element_symbol"]
        self.atoms = self.entity.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy()

    def get_centers(self):
        raise NotImplementedError()

    def get_radii(self):
        raise NotImplementedError()