r"""Definition of the Model and PDBEntity class."""
from __future__ import annotations
from typing import Sequence
import re
from copy import deepcopy

from biopandas.pdb import PandasPdb as pd
from biopandas.pdb.pandas_pdb import PandasPdb
from Bio.PDB.Structure import Structure
import numpy as np
from numpy.typing import NDArray

from sadic.utils import Repr


class Model(Repr):
    r"""A class representing a model of a pdb protein model.

    Attributes:
        atom_indexes (NDArray[np.int32]):
            The indexes of the atoms in the original pdb file.
        
        atom_types (NDArray[np.string_]):
            The types of the atoms in the model.
        
        atoms (NDArray[np.float32]):
            The coordinates of the atoms in the model.
        
        last_probe_radius (float | None):
            The last probe radius used to calculate the radii of the atoms.
        
        radii (NDArray[np.float32] | None):
            The Van Der Waals radii of the atoms in the model.
    """

    def __init__(self, pdb_entity: PandasPdb, model_idx: int):
        r"""Builds the Model object based on the PDBEntity object given in input.

        Args:
            pdb_entity (PDBEntity):
                The PDBEntity object containing the original pdb file.
            
            model_idx (int):
                The index of the model to be used to build the Model object.
        """
        self.model: PandasPdb | None = None
        self.code: str
        self.atom_indexes: NDArray[np.int32]
        self.atom_types: NDArray[np.string_]
        self.atoms: NDArray[np.float32]
        self.radii: NDArray[np.float32] | None = None
        self.last_probe_radius: float | None = None

        self.build(pdb_entity, model_idx)

    def build(self, pdb_entity: PandasPdb, model_idx: int):
        r"""Builds the Model object based on the PDBEntity object given in input.

        Args:
            pdb_entity (PDBEntity):
                The PDBEntity object containing the original pdb file.
            
            model_idx (int):
                The index of the model to be used to build the Model object.
        """
        self.model = pdb_entity.get_model(model_idx)
        self.code = pdb_entity.code
        self.atom_indexes = self.model.df["ATOM"]["atom_number"].to_numpy()
        self.atom_types = self.model.df["ATOM"]["element_symbol"].to_numpy()
        self.atoms = self.model.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy()

    def get_atom_indexes(self) -> NDArray[np.int32]:
        r"""Returns the indexes of the atoms in the Model object.

        Returns (NDArray[np.int32]):
            The indexes of the atoms in the Model object.
        """
        return self.atom_indexes

    def get_centers(self) -> NDArray[np.float32]:
        r"""Returns the coordinates of the atoms in the PDBEntity object.

        Returns (NDArray[np.float32]):
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

        Returns (NDArray[np.float32]):
            The radii of the atoms in the PDBEntity object, increased by the probe radius.
        """
        final_probe_radius: float = (
            probe_radius if probe_radius is not None else PDBEntity.vdw_radii['O']
        )

        if self.radii is None or self.last_probe_radius != final_probe_radius:
            self.last_probe_radius = final_probe_radius
            self.radii = np.array(
                [
                    PDBEntity.vdw_radii[atom_type] + final_probe_radius
                    for atom_type in self.atom_types
                ],
                dtype=np.float32,
            )

        return self.radii

    def __len__(self) -> int:
        r"""Returns the number of atoms in the Model object.

        Returns (int):
            The number of atoms in the Model object.
        """
        return len(self.atoms)

    def filter_(
        self,
        filter_by: dict[str, str | int | Sequence[str] | Sequence[int]]
        | NDArray[np.float32]
        | tuple[NDArray[np.float32], float],
    ) -> None:
        r"""Filters the Model object inplace, based on the filter given in input.

        Args:
            filter_by (None | dict[str, str | int | Sequence[str] | Sequence[int]]
            | tuple[NDArray[np.float32], float] | NDArray[np.float32]):
                The filter to apply to the atoms of the protein. If None, no filter is applied. If a
                tuple of a np.ndarray and a float, they are assumed to represent the center and
                radius of a sphere and only the atoms inside the sphere are considered. If a numpy
                array, it must be an array with shape (n_points, 3) containing the coordinates of
                the points to be selected. If a dictionary, the keys are the columns of the PDB file
                and the values are the values to select. The keys can be one of the following:
                "atom_number", "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The values can be a single value or a list of values. Defaults to
                None.
        """
        if self.model is None:
            raise ValueError("No model to filter")

        if isinstance(filter_by, np.ndarray):
            if filter_by.ndim != 2 or filter_by.shape[1] != 3:
                raise ValueError(
                    "The array given in input must be a valid array of coordinates with shape "
                    "(n_points, 3)."
                )
            self.atom_indexes = np.arange(filter_by.shape[0], dtype=np.int32)
            self.atom_types = np.full(filter_by.shape[0], "X", dtype=np.string_)
            self.atoms = filter_by

            return

        if isinstance(filter_by, dict):
            for column, value in filter_by.items():
                values = value if isinstance(value, Sequence) else [value]
                self.model.df["ATOM"] = self.model.df["ATOM"][
                    self.model.df["ATOM"][column].isin(values)
                ]
        elif isinstance(filter_by, tuple):
            if len(filter_by) != 2:
                raise ValueError(
                    "The tuple given in input must contain the center and radius of the sphere."
                )
            center, radius = filter_by
            if not isinstance(center, np.ndarray) or center.ndim != 1 or center.shape[0] != 3:
                raise ValueError(
                    "The center of the sphere must be a valid array of coordinates with shape "
                    "(3,)."
                )
            if not isinstance(radius, float):
                raise ValueError("The radius of the sphere must be a valid float.")

            self.model.df["ATOM"] = self.model.df["ATOM"][
                np.linalg.norm(
                    self.model.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy() - center,
                    axis=1,
                )
                <= radius
            ]

        self.atom_indexes = self.model.df["ATOM"]["atom_number"].to_numpy()
        self.atom_types = self.model.df["ATOM"]["element_symbol"].to_numpy()
        self.atoms = self.model.df["ATOM"][["x_coord", "y_coord", "z_coord"]].to_numpy()

    def filter(
        self,
        filter_arg: dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32],
    ) -> Model:
        r"""Filters the Model object, based on the filter given in input.

        Args:
            filter_by (None | dict[str, str | int | Sequence[str] | Sequence[int]]
            | tuple[NDArray[np.float32], float] | NDArray[np.float32]):
                The filter to apply to the atoms of the protein. If None, no filter is applied. If a
                tuple of a np.ndarray and a float, they are assumed to represent the center and
                radius of a sphere and only the atoms inside the sphere are considered. If a numpy
                array, it must be an array with shape (n_points, 3) containing the coordinates of
                the points to be selected. If a dictionary, the keys are the columns of the PDB file
                and the values are the values to select. The keys can be one of the following:
                "atom_number", "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The values can be a single value or a list of values. Defaults to
                None.

        Returns (Model):
            The filtered Model object.
        """
        model = deepcopy(self)
        model.filter_(filter_arg)
        return model


class PDBEntity(Repr):
    r"""A class representing a pdb protein structure or set of structures.

    Attributes:
        nmodels (int):
            The number of models in the PDBEntity object.
        
        entity (PandasPdb):
            The biopandas.pdb.PandasPdb object representing the pdb structure.
        
        models (dict[int, Model]):
            The list of models in the PDBEntity object. For consistency with the pdb notation, model
            indexes start from 1.
        
        last_probe_radius (float | None):
            The last probe radius used to calculate the radii of the atoms.


    Class Attributes:
        vdw_radii_1939 (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, as defined in the
            1939 paper by Pauling.
        
        vdw_radii_1964 (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, as defined in the
            1964 paper by Bondi.
        
        vdw_radii_1996 (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, as defined in the
            1996 paper by Rowland.
        
        vdw_radii_2004 (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, as defined in the
            2004 sadic version (they are the same as the 1964 ones).
        
        vdw_radii (dict[str, float]):
            The Van Der Waals radii of the atoms that are present in proteins, to be used by
            default.

    Class Methods:
        __init__ (arg: str | PandasPdb | Structure, mode: str = "infer") -> None:
            Builds the PDBEntity object based on the protein given in arg.
        
        build (arg: str | PandasPdb | Structure, mode: str = "infer") -> None:
            Builds the PDBEntity object based on the protein given in arg.
        
        complete_build_from_entity (self) -> None:
            Completes the build of the PDBEntity object based on the entity attribute.
        
        build_from_biopandas (self, arg: PandasPdb) -> None:
            Builds the PDBEntity object based on the biopandas.pdb.PandasPdb object given in arg.
        
        build_from_biopython (self, arg: Structure) -> None:
            Builds the PDBEntity object based on the biopython.structure.Structure object given in
            arg.
        
        build_from_pdb_file (self, arg: str) -> None:
            Builds the PDBEntity object based on the pdb file given in arg.
        
        build_from_gzpdb_file (self, arg: str) -> None:
            Builds the PDBEntity object based on the gzipped pdb file given in arg.
        
        build_from_url (self, url: str) -> None:
            Builds the PDBEntity object based on the url to a pdb structure page given in arg.
        
        build_from_code (self, code: str) -> None:
            Builds the PDBEntity object based on the code of a pdb structure given in arg.
        
        __len__ (self) -> int:
            Returns the number of atoms in the PDBEntity object.
        
        get_centers (self) -> NDArray[np.float32]:
            Returns the coordinates of the atoms in the PDBEntity object.
        
        get_radii (self, probe_radius: float | None = None) -> NDArray[np.float32]:
            Returns the Van Der Waals radii of the atoms in the PDBEntity object.
    """

    _pdb_code_regex: str = r"[1-9][a-zA-Z0-9]{3}"
    _pdb_url_regex: str = r"(http[s]?://)?www.rcsb.org/structure/" + _pdb_code_regex + r"[/]?"

    vdw_radii_1938: dict[str, float] = {"H": 1.2, "C": 1.7, "N": 1.5, "O": 1.4, "S": 1.85}
    vdw_radii_1964: dict[str, float] = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}
    vdw_radii_1996: dict[str, float] = {"H": 1.1, "C": 1.77, "N": 1.64, "O": 1.58, "S": 1.81}
    vdw_radii_2004: dict[str, float] = vdw_radii_1938
    vdw_radii: dict[str, float] = vdw_radii_2004

    def __init__(
        self,
        arg: str | PandasPdb | Structure,
        mode: str = "infer",
        vdw_radii: None | dict[str, float] = None,
    ) -> None:
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
            
            vdw_radii (None | dict[str, float], optional):
                The Van Der Waals radii of the atoms that are present in proteins. Atom types that
                are not present in the dictionary will be assigned the default radii. Atom types
                that are not present in proteins will be ignored. If None, the default radii will be
                used. Defaults to None.

        Raises:
            TypeError: Raised when arg is not a string, biopandas.pdb.PandasPdb object or
                biopython.structure.Structure object.
            
            ValueError: Raised when the mode is to be inferred and arg is not a valid string or
                when the vdw_radii argument is not None but the key-value pairs are not valid Van
                Der Waals radii values.
        """
        self.nmodels: int
        self.entity: PandasPdb
        self.code: str
        self.models: dict[int, Model] = {}
        self.last_probe_radius: float | None = None

        if vdw_radii is not None:
            for atom_type in PDBEntity.vdw_radii:
                if atom_type not in vdw_radii:
                    raise ValueError(f"Atom type {atom_type} not found in vdw_radii.")

            for atom_type in vdw_radii:
                if atom_type not in PDBEntity.vdw_radii:
                    raise ValueError(f"Atom type {atom_type} not found in vdw_radii.")

                if vdw_radii[atom_type] <= 0.0:
                    raise ValueError(
                        f"Atom Van Der Waals radius for atom type {atom_type} must be positive "
                        f"(got {vdw_radii[atom_type]})."
                    )

                PDBEntity.vdw_radii[atom_type] = vdw_radii[atom_type]

        self.build(arg, mode)

    def build(
        self, arg: str | PandasPdb | Structure, mode: str = "infer", max_nmodels: int = 1000
    ) -> None:
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
            
            max_nmodels (int, optional):
                The maximum number of models to be loaded. It is a security measure to prevent
                too long loops. Defaults to 1000.

        Raises:
            TypeError:
                Raised when arg is not a string, biopandas.pdb.PandasPdb object or
                biopython.structure.Structure object.
            
            ValueError:
                Raised when the mode is to be inferred and arg is not a valid string.
        """
        # TO DO: riscrivere tutti questi branch usando un dizionario
        if mode == "biopandas":
            if not isinstance(arg, PandasPdb):
                raise TypeError("arg must be a biopandas.pdb.PandasPdb object.")
            self.build_from_biopandas(arg, max_nmodels)
        elif mode == "biopython":
            if not isinstance(arg, Structure):
                raise TypeError("arg must be a biopython.structure.Structure object.")
            self.build_from_biopython(arg, max_nmodels)
        elif mode == "pdb":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the path of a pdb file.")
            self.build_from_pdb_file(arg, max_nmodels)
        elif mode == "gz":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the path of a gzipped pdb file.")
            self.build_from_gzpdb_file(arg, max_nmodels)
        elif mode == "url":
            if not isinstance(arg, str):
                raise TypeError(
                    "arg must be a string representing the url to a pdb structure page."
                )
            self.build_from_url(arg, max_nmodels)
        elif mode == "code":
            if not isinstance(arg, str):
                raise TypeError("arg must be a string representing the code of a pdb structure.")
            self.build_from_code(arg, max_nmodels)
        elif mode == "infer":
            if isinstance(arg, PandasPdb):
                self.build(arg, "biopandas", max_nmodels)
            elif isinstance(arg, Structure):
                self.build(arg, "biopython", max_nmodels)
            elif isinstance(arg, str):
                if arg.endswith(".pdb"):
                    self.build(arg, "pdb", max_nmodels)
                elif arg.endswith(".gz"):
                    self.build(arg, "gz", max_nmodels)
                elif re.match(PDBEntity._pdb_url_regex, arg):
                    self.build(arg, "url", max_nmodels)
                elif re.match(PDBEntity._pdb_code_regex, arg):
                    self.build(arg, "code", max_nmodels)
                else:
                    raise ValueError(
                        "arg must be a pdb file in .pdb or .gz or a url to a structure page or a "
                        "code of a pdb structure."
                    )
            else:
                raise TypeError(
                    "arg must be a string, biopandas.pdb.PandasPdb object or"
                    " biopython.structure.Structure object."
                )
        else:
            raise ValueError(
                "mode must be one of 'biopandas', 'biopython', 'pdb', 'gz', 'url', 'code', or "
                "'infer'."
            )

    def complete_build_from_entity(self, max_nmodels: int = 1000) -> None:
        r"""Completes the building of the PDBEntity object using the assigned PandasPdb entity.

        For each model composing the pdb file, creates a Model object and adds it to the models
        attribute. Also initializes the last_probe_radius attribute to None.

        Args:
            max_nmodels (int, optional):
                The maximum number of models to be loaded. It is a security measure to prevent
                too long loops. Defaults to 1000.
        """
        self.code = self.entity.df["OTHERS"]["entry"][0][-4:]
        for model_index in range(1, max_nmodels + 1):
            if self.entity.get_model(model_index).df["ATOM"].empty:
                break

            self.models[model_index] = Model(self.entity, model_index)

        self.nmodels = len(self.models)

    def build_from_biopandas(self, arg: PandasPdb, max_nmodels: int = 1000) -> None:
        r"""Builds the PDBEntity object based on the PandasPdb given in input.

        Writes the PandasPdb object to the entity attribute and calls the complete_build_from_entity
        method.

        Args:
            arg (PandasPdb):
                The PandasPdb object to be used to build the PDBEntity object.
            
            max_nmodels (int, optional):
                The maximum number of models to be loaded. It is a security measure to prevent
                too long loops. Defaults to 1000.
        """
        self.entity = arg
        self.complete_build_from_entity(max_nmodels)

    def build_from_biopython(self, arg: Structure, max_nmodels: int = 1000) -> None:
        r"""Builds the PDBEntity object based on the biopython.structure.Structure give in input.

        To be implemented.
        """
        raise NotImplementedError

    def build_from_pdb_file(self, path: str, max_nmodels: int = 1000) -> None:
        r"""Builds the PDBEntity object based on the plain pdb file (.pdb) given in input.

        Args:
            path (str):
                The path to a pdb file to be used to build the PDBEntity object.
        """
        if not path.endswith(".pdb"):
            raise ValueError("path must be a string representing the path of a pdb file.")

        self.entity: PandasPdb = pd().read_pdb(path)
        self.complete_build_from_entity(max_nmodels)

    def build_from_gzpdb_file(self, path: str, max_nmodels: int = 1000) -> None:
        r"""Builds the PDBEntity object based on the gzipped pdb file (.gz) given in input.

        Args:
            path (str):
                The path to a gzipped pdb file to be used to build the PDBEntity object.
        """
        if not path.endswith(".pdb.gz"):
            raise ValueError("path must be a string representing the path of a pdb file.")

        self.entity: PandasPdb = pd().read_pdb(path)
        self.complete_build_from_entity(max_nmodels)

    def build_from_url(self, url: str, max_nmodels: int = 1000) -> None:
        r"""Builds the PDBEntity object based on the url to a pdb structure page given in input.

        Extracts the code of the pdb structure from the url and builds the PDBEntity object based on
        the code.

        Args:
            url (str):
                The url to a pdb structure page to be used to build the PDBEntity object.
            
            max_nmodels (int, optional):
                The maximum number of models to be loaded. It is a security measure to prevent
                too long loops. Defaults to 1000.

        Raises:
            ValueError: Raised when the url is not a valid url to a pdb structure page.
        """
        if not re.match(PDBEntity._pdb_url_regex, url):
            raise ValueError(
                f"url must be an url to a page of a pdb structure (format "
                f"{PDBEntity._pdb_url_regex})."
            )

        stripped_url: str = url.rstrip("/")
        code: str = stripped_url[-4:]

        self.build_from_code(code, max_nmodels)

    def build_from_code(self, code: str, max_nmodels: int = 1000) -> None:
        r"""Builds the PDBEntity object based on the code of a pdb structure given in input.

        Uses the biopandas.pdb.fetch_pdb method to fetch the pdb structure from the rcsb pdb
        database and builds the PDBEntity object based on the fetched structure.

        Args:
            code (str):
                The code of a pdb structure to be used to build the PDBEntity object.
            
            max_nmodels (int, optional):
                The maximum number of models to be loaded. It is a security measure to prevent
                too long loops. Defaults to 1000.

        Raises:
            ValueError: Raised when the code is not a valid code of a pdb structure.
        """
        if not re.match(PDBEntity._pdb_code_regex, code):
            raise ValueError(
                f"code must be a code representing a pdb structure (format "
                f"{PDBEntity._pdb_code_regex})."
            )

        self.entity: PandasPdb = pd().fetch_pdb(code)
        self.complete_build_from_entity(max_nmodels)

    def __len__(self) -> int:
        r"""Returns the number of atoms in the PDBEntity object.

        Returns (int):
            The number of atoms in the PDBEntity object.
        """
        atom_indexes: list[NDArray[np.int32]] = []
        for model in self.models.values():
            atom_indexes.append(model.get_atom_indexes())

        return len(np.unique(np.concatenate(atom_indexes)))

    def filter_(
        self,
        filter_by: dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32],
    ) -> None:
        r"""Filters the Models of the PDBEntity object inplace, based on the filter given in input.

        Args:
            filter_by (None | dict[str, str | int | Sequence[str] | Sequence[int]]
            | tuple[NDArray[np.float32], float] | NDArray[np.float32]):
                The filter to apply to the atoms of the protein. If None, no filter is applied. If a
                tuple of a np.ndarray and a float, they are assumed to represent the center and
                radius of a sphere and only the atoms inside the sphere are considered. If a numpy
                array, it must be an array with shape (n_points, 3) containing the coordinates of
                the points to be selected. If a dictionary, the keys are the columns of the PDB file
                and the values are the values to select. The keys can be one of the following:
                "atom_number", "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The values can be a single value or a list of values. Defaults to
                None.
        """
        for model in self.models.values():
            model.filter_(filter_by)

    def filter(
        self,
        filter_arg: dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32],
    ) -> PDBEntity:
        r"""Filters the Models of the PDBEntity object, based on the filter given in input.

        Args:
            filter_by (None | dict[str, str | int | Sequence[str] | Sequence[int]]
            | tuple[NDArray[np.float32], float] | NDArray[np.float32]):
                The filter to apply to the atoms of the protein. If None, no filter is applied. If a
                tuple of a np.ndarray and a float, they are assumed to represent the center and
                radius of a sphere and only the atoms inside the sphere are considered. If a numpy
                array, it must be an array with shape (n_points, 3) containing the coordinates of
                the points to be selected. If a dictionary, the keys are the columns of the PDB file
                and the values are the values to select. The keys can be one of the following:
                "atom_number", "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The values can be a single value or a list of values. Defaults to
                None.

        Returns (Entity):
            The filtered PDBEntity object.
        """
        entity = deepcopy(self)
        entity.filter_(filter_arg)
        return entity
