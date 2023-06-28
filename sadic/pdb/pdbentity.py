r"""Definition of the PDBEntity class."""

import re

from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure
import numpy as np
from numpy.typing import NDArray


class PDBEntity:
    r"""A class representing a pdb protein structure or set of structures.

    Attributes:
        entity (PandasPdb):
            The biopandas.pdb.PandasPdb object representing the pdb structure.
        atom_types (NDArray[np.string_]):
            The types of the atoms in the structure.
        atoms (NDArray[np.float32]):
            The coordinates of the atoms in the structure.
        last_probe_radius (float | None):
            The last probe radius used to calculate the radii of the atoms.
        radii (NDArray[np.float32] | None):
            The Van Der Waals radii of the atoms in the structure.

    Class Attributes:
        vdw_radii_2004 (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, updated in 2004.
        vdw_radii_2023 (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, updated in 2023.
        vdw_radii (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, to be used by
            default.
    """

    _pdb_code_regex: str = r"[1-9][a-zA-Z0-9]{3}"
    _pdb_url_regex: str = r"(http[s]?://)?www.rcsb.org/structure/" + _pdb_code_regex + r"[/]?"

    _vdw_radii_2004: dict[str, float] = {"C": 1.7, "N": 1.5, "O": 1.4, "S": 1.85}

    _vdw_radii_2023: dict[str, float] = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}

    _vdw_radii: dict[str, float] = _vdw_radii_2004

    def __init__(self, arg: str | PandasPdb | Structure, mode: str = "infer") -> None:
        r"""Builds the PDBEntity object based on the protein given in arg.

        The protein can be given in one of the following ways:
        1. A biopandas.pdb.PandasPdb object.
        2. A biopython.structure.Structure object.
        3. A string representing the path of a pdb file.
        4. A string representing the path of a gzipped pdb file.
        5. A string representing the url to a pdb structure page.
        6. A string representing the code of a pdb structure.
        If the mode is "infer", the mode will be inferred from the type of arg. If the mode is not
        "infer", the mode will be used to determine how to build the PDBEntity object.

        Args:
            arg (str | PandasPdb | Structure):
                The protein to be represented by the PDBEntity object.
            mode (str, optional):
                The mode to be used to build the PDBEntity object. Can be one of "biopandas",
                "biopython", "pdb", "gz", "url", "code" or "infer". Defaults to "infer".

        Raises:
            TypeError: Raised when arg is not a string, biopandas.pdb.PandasPdb object or
                biopython.structure.Structure object.
            ValueError: Raised when the mode is to be inferred and arg is not a valid string.
        """
        self.entity: PandasPdb
        self.atom_types: NDArray[np.string_]
        self.atoms: NDArray[np.float32]
        self.last_probe_radius: float | None
        self.radii: NDArray[np.float32] | None

        self.build(arg, mode)

    def build(self, arg: str | PandasPdb | Structure, mode: str = "infer") -> None:
        r"""Builds the PDBEntity object based on the protein given in arg.

        The protein can be given in one of the following ways:
        1. A biopandas.pdb.PandasPdb object.
        2. A biopython.structure.Structure object.
        3. A string representing the path of a pdb file.
        4. A string representing the path of a gzipped pdb file.
        5. A string representing the url to a pdb structure page.
        6. A string representing the code of a pdb structure.
        If the mode is "infer", the mode will be inferred from the type of arg. If the mode is not
        "infer", the mode will be used to determine how to build the PDBEntity object.

        Args:
            arg (str | PandasPdb | Structure):
                The protein to be represented by the PDBEntity object.
            mode (str, optional):
                The mode to be used to build the PDBEntity object. Can be one of "biopandas",
                "biopython", "pdb", "gz", "url", "code" or "infer". Defaults to "infer".

        Raises:
            TypeError:
                Raised when arg is not a string, biopandas.pdb.PandasPdb object or
                biopython.structure.Structure object.
            ValueError:
                Raised when the mode is to be inferred and arg is not a valid string.
        """
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
                self.build(arg, "biopandas")
            elif isinstance(arg, Structure):
                self.build(arg, "biopython")
            elif isinstance(arg, str):
                if arg.endswith(".pdb"):
                    self.build(arg, "pdb")
                elif arg.endswith(".gz"):
                    self.build(arg, "gz")
                elif re.match(PDBEntity._pdb_url_regex, arg):
                    self.build(arg, "url")
                elif re.match(PDBEntity._pdb_code_regex, arg):
                    self.build(arg, "code")
                else:
                    raise ValueError(
                        "arg must be a pdb file in .pdb or .gz or a url to a structure page or a "
                        "code of a pdb structure"
                    )
            else:
                raise TypeError(
                    "arg must be a string, biopandas.pdb.PandasPdb object or"
                    " biopython.structure.Structure object"
                )
        else:
            raise ValueError(
                "mode must be one of 'biopandas', 'biopython', 'pdb', 'gz', 'url', 'code', or "
                "'infer'"
            )

    def complete_build_from_entity(self) -> None:
        r"""Completes the building of the PDBEntity object using the assigned PandasPdb entity.

        Extracts the atom types and coordinates from the PandasPDB entity and writes it to the
        atom_types and atoms attributes respectively. Also initializes the last_probe_radius and
        radii attributes to None.
        """
        self.atom_types = self.entity.df["ATOM"]["element_symbol"].to_numpy()
        self.atoms = self.entity.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy(
            dtype=np.float32
        )
        self.last_probe_radius = None
        self.radii = None

    def build_from_biopandas(self, arg: PandasPdb) -> None:
        r"""Builds the PDBEntity object based on the PandasPdb given in input.

        Writes the PandasPdb object to the entity attribute and calls the complete_build_from_entity
        method.

        Args:
            arg (PandasPdb):
                The PandasPdb object to be used to build the PDBEntity object.
        """
        self.entity = arg
        self.complete_build_from_entity()

    def build_from_biopython(self, arg: Structure) -> None:
        r"""Builds the PDBEntity object based on the biopython.structure.Structure give in input.

        To be implemented.
        """
        raise NotImplementedError

    def build_from_pdb_file(self, arg: str) -> None:
        r"""Builds the PDBEntity object based on the plain pdb file (.pdb) given in input.

        To be implemented.
        """
        raise NotImplementedError

    def build_from_gzpdb_file(self, arg: str) -> None:
        r"""Builds the PDBEntity object based on the gzipped pdb file (.gz) given in input.

        To be implemented.
        """
        raise NotImplementedError

    def build_from_url(self, url: str) -> None:
        r"""Builds the PDBEntity object based on the url to a pdb structure page given in input.

        Extracts the code of the pdb structure from the url and builds the PDBEntity object based on
        the code.

        Args:
            url (str):
                The url to a pdb structure page to be used to build the PDBEntity object.

        Raises:
            ValueError: Raised when the url is not a valid url to a pdb structure page.
        """
        if not re.match(PDBEntity._pdb_url_regex, url):
            raise ValueError(
                f"url must be an url to a page of a pdb structure (format "
                f"{PDBEntity._pdb_url_regex})"
            )

        stripped_url: str = url.rstrip("/")
        code: str = stripped_url[-4:]

        self.build_from_code(code)

    def build_from_code(self, code: str) -> None:
        r"""Builds the PDBEntity object based on the code of a pdb structure given in input.

        Uses the biopandas.pdb.fetch_pdb method to fetch the pdb structure from the rcsb pdb
        database and builds the PDBEntity object based on the fetched structure.

        Args:
            code (str):
                The code of a pdb structure to be used to build the PDBEntity object.

        Raises:
            ValueError: Raised when the code is not a valid code of a pdb structure.
        """
        if not re.match(PDBEntity._pdb_code_regex, code):
            raise ValueError(
                f"code must be a code representing a pdb structure (format "
                f"{PDBEntity._pdb_code_regex})"
            )

        self.entity: PandasPdb = pd().fetch_pdb(code)
        self.complete_build_from_entity()

    def __len__(self) -> int:
        r"""Returns the number of atoms in the PDBEntity object.

        Returns:
            int:
                The number of atoms in the PDBEntity object.
        """
        return len(self.atoms)

    def get_centers(self) -> NDArray[np.float32]:
        r"""Returns the coordinates of the atoms in the PDBEntity object.

        Returns:
            NDArray[np.float32]:
                The coordinates of the atoms in the PDBEntity object.
        """
        return self.atoms

    def get_radii(self, probe_radius: float | None = None) -> NDArray[np.float32]:
        r"""Returns the radii of the atoms in the PDBEntity object.

        Uses the probe_radius given in input to increase the atoms' radii, to account for the size
        of potential probe atoms. Uses the argument probe_radius if not None, otherwise uses the
        minimum Van Der Waals radius of the building atoms of proteins.

        Args:
            probe_radius (float, optional):
                The radius of the test atoms to be used to increase the atoms' radii. Defaults to
                None.

        Returns:
            NDArray[np.float32]:
                The radii of the atoms in the PDBEntity object, increased by the probe radius.
        """
        final_probe_radius: float = (
            probe_radius if probe_radius is not None else min(PDBEntity._vdw_radii.values())
        )

        if self.radii is None or self.last_probe_radius != final_probe_radius:
            self.last_probe_radius = final_probe_radius
            self.radii = np.array(
                [
                    PDBEntity._vdw_radii[atom_type] + final_probe_radius
                    for atom_type in self.atom_types
                ],
                dtype=np.float32,
            )

        return self.radii
