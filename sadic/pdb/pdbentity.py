from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure
import numpy as np
import re

from numpy.typing import NDArray

class PDBEntity:
    _pdb_code_regex = r"[1-9][a-zA-Z0-9]{3}"
    _pdb_url_regex = r"(http[s]?://)?www.rcsb.org/structure/" + _pdb_code_regex + r"[/]?"

    vdw_radii_2004 = {
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

    vdw_radii = vdw_radii_2004

    def __init__(self, arg: str | PandasPdb | Structure, mode: str = "infer") -> None:
        if mode == "biopandas":
            if not isinstance(arg, PandasPdb):
                raise TypeError("arg must be a biopandas.pdb.PandasPdb object")
            self.build_from_biopandas(arg)

        elif mode == "biopython":
            if not isinstance(arg, Structure):
                raise TypeError("arg must be a biopython.structure.Structure object")
            self.build_from_biopython(arg)
        
        elif mode == "pdb":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the path of a pdb file")
            self.build_from_pdb_file(arg)
        
        elif mode == "gz":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the path of a gzipped pdb file")
            self.build_from_gzpdb_file(arg)
        
        elif mode == "url":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the url to a pdb structure page")
            self.build_from_url(arg)
        
        elif mode == "code":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the code of a pdb structure")
            self.build_from_code(arg)
        
        elif mode == "infer":
            if isinstance(arg, PandasPdb):
                self.__init__(arg, "biopandas")
        
            elif isinstance(arg, Structure):
                self.__init__(arg, "biopython")
        
            elif isinstance(arg, str):
                if arg.endswith('.pdb'):
                    self.__init__(arg, "pdb")
        
                elif arg.endswith('.gz'):
                    self.__init__(arg, "gz")
        
                elif re.match(PDBEntity._pdb_url_regex, arg):
                    self.__init__(arg, "url")
        
                elif re.match(PDBEntity._pdb_code_regex, arg):
                    self.__init__(arg, "code")
        
                else:
                    raise ValueError(f"arg must be a pdb file in .pdb or .gz or a url to a pdb structure page or a code of a pdb structure")

            else:
                raise TypeError(f"arg must be a string, biopandas.pdb.PandasPdb object or biopython.structure.Structure object")

        else:
            raise ValueError("mode must be one of 'biopandas', 'biopython', 'pdb', 'gz', 'url', 'code', or 'infer'")

    def complete_build_from_entity(self) -> None:
        self.atom_types: NDArray[np.string_] = self.entity.df["ATOM"]["element_symbol"].to_numpy()
        self.atoms: NDArray[np.float32] = self.entity.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)
        self.last_probe_radius: float | None = None
        self.radii: NDArray[np.float32] | None = None

    def build_from_biopandas(self, arg: PandasPdb) -> None:
        self.entity = arg
        self.complete_build_from_entity()

    def build_from_biopython(self, arg: Structure) -> None:
        raise NotImplementedError

    def build_from_pdb_file(self, arg: str) -> None:
        raise NotImplementedError

    def build_from_gzpdb_file(self, arg: str) -> None:
        raise NotImplementedError

    def build_from_url(self, url: str) -> None:
        if not re.match(PDBEntity._pdb_url_regex, url):
            raise ValueError(f"url must be an url to a page of a pdb structure (format {PDBEntity._pdb_url_regex})")
        
        stripped_url: str = url.rstrip("/")
        code: str = stripped_url[-4:]
        
        self.build_from_code(code)

    def build_from_code(self, code: str) -> None:
        if not re.match(PDBEntity._pdb_code_regex, code):
            raise ValueError(f"code must be a code representing a pdb structure (format {PDBEntity._pdb_code_regex})")
        
        self.entity: PandasPdb = pd().fetch_pdb(code)
        self.complete_build_from_entity()

    def __len__(self) -> int:
        return len(self.atoms)
    
    def get_centers(self) -> NDArray[np.float32]:
        return self.atoms

    def get_radii(self, probe_radius: float | None = None) -> NDArray[np.float32]:
        final_probe_radius: float = probe_radius if probe_radius is not None else min(PDBEntity.vdw_radii.values())
            
        if self.radii is None or self.last_probe_radius != final_probe_radius:
            self.last_probe_radius = final_probe_radius
            self.radii = np.array([PDBEntity.vdw_radii[atom_type] + final_probe_radius for atom_type in self.atom_types], dtype=np.float32)

        return self.radii