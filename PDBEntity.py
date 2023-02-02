from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb
import numpy as np

from numpy.typing import NDArray

class PDBEntity:
    vdw_radii_2004 = {
        'H': 1.2,
        'C': 1.7,
        'N': 1.5,
        'O': 1.4,
        'S': 1.85,
    }

    vdw_radii_2023 = {
        "C": 1.7,
        "N": 1.55,
        "O": 1.52,
        "S": 1.8,
    }

    vdw_radii = vdw_radii_2023

    def __init__(self, id: str) -> None:
        self.entity: PandasPdb = pd().fetch_pdb(id)
        self.atom_types: NDArray[np.string_] = self.entity.df["ATOM"]["element_symbol"].to_numpy(dtype=np.string_)
        self.atoms: NDArray[np.float32] = self.entity.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
        self.last_probe_radius: float | None = None
        self.radii: NDArray[np.float32] | None = None

    def get_centers(self) -> NDArray[np.float32]:
        return self.atoms

    def get_radii(self, probe_radius: float | None = None) -> NDArray[np.float32]:
        final_probe_radius: float = probe_radius if probe_radius is not None else min(PDBEntity.vdw_radii.values())

        if probe_radius == self.last_probe_radius and self.radii is not None:
            return self.radii

        return np.array([PDBEntity.vdw_radii[atom_type] + final_probe_radius for atom_type in self.atom_types], dtype=np.float32)