r"""Result of the SADIC algorithm for a single model of a protein and for multiple models."""

from __future__ import annotations
from typing import Any, Sequence
from copy import deepcopy
import os

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from sadic.pdb import PDBEntity, Model
from sadic.utils import Repr


class SadicModelResult(Repr):
    r"""Result of the SADIC algorithm for a single model of a protein.

    Attributes:
        atom_index (NDArray[np.int32]):
            The index of the atoms/residues of the protein.
        depth_index (NDArray[np.float32]):
            The SADIC depth index of the atoms of the protein.

    Methods:
        __init__:
            Initialize the result of the SADIC algorithm.
    """

    def __init__(
        self,
        atom_index: NDArray[np.int32],
        depth_index: NDArray[np.float32],
        model: Model,
    ) -> None:
        r"""Initialize the result of the SADIC algorithm.

        Args:
            atom_index (NDArray[np.int32]):
                The index of the atoms of the protein.
            depth_index (NDArray[np.float32]):
                The SADIC depth index of the atoms of the protein.
            model (Model):
                The entity of the protein model, containing only the atoms used to compute the
                SADIC depth index.
        """
        self.atom_index: NDArray[np.int32]
        self.depth_index: NDArray[np.float32]
        self.model: Model

        self.build(atom_index, depth_index, model)

    def build(
        self, atom_index: NDArray[np.int32], depth_index: NDArray[np.float32], model: Model
    ) -> None:
        r"""Build the result of the SADIC algorithm from the atoms of the protein.

        Args:
            atom_index (NDArray[np.int32]):
                The index of the atoms of the protein.
            depth_index (NDArray[np.float32]):
                The SADIC depth index of the atoms of the protein.
            model (Model):
                The entity of the protein model, containing only the atoms used to compute the
                SADIC depth index.
        """
        self.atom_index = atom_index
        self.depth_index = depth_index
        self.model = model

    def get_depth_index(
        self,
        filter_by: None
        | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32] = None,
        get_index: bool = False,
    ):
        r"""Return the depth index of the atoms of the protein.

        Can return the depth index of the atoms of the protein, filtered by a filter argument.

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
            get_index (bool):
                If True, also return the index of the atoms/residue of the protein. Defaults to
                False.

        Returns (NDArray[np.float32] | tuple[NDArray[np.int32], NDArray[np.float32]]):
            The depth index of the atoms of the protein.
        """
        output_atom_index: NDArray[np.int32] = self.atom_index
        output_depth_index: NDArray[np.float32] = self.depth_index

        if len(self.model.atom_types) != 0:
            if self.model.atom_types[0] != "X" and filter_by is not None:
                filtered_model: Model = self.model.filter(filter_by)
                if filtered_model.model is None:
                    raise ValueError("The filter is not valid.")

                filter_index = np.where(
                    np.isin(self.atom_index, filtered_model.model.df["ATOM"]["atom_number"].to_numpy())
                )[0]

                output_atom_index = self.atom_index[filter_index]
                output_depth_index = self.depth_index[filter_index]

        return output_atom_index, output_depth_index if get_index else output_depth_index

    def filter_(
        self,
        filter_by: None
        | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32] = None,
    ) -> None:
        r"""Filter (inplace) the entity inside the result object.

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
        if filter_by is not None and self.model.atom_types[0] != "X":
            self.model.filter_(filter_by)

            filtered_model: Model = self.model.filter(filter_by)
            if filtered_model.model is None:
                raise ValueError("The filter is not valid.")

            filter_index = np.where(
                np.isin(self.atom_index, filtered_model.model.df["ATOM"]["atom_number"].to_numpy())
            )[0]

            self.atom_index = self.atom_index[filter_index]
            self.depth_index = self.depth_index[filter_index]

    def filter(
        self,
        filter_by: None
        | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32] = None,
    ) -> SadicModelResult:
        r"""Filter the entity inside the result object according to the filter_by argument.

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
        if filter_by is None:
            return self

        result_copy: SadicModelResult = deepcopy(self)
        result_copy.filter_(filter_by)
        return result_copy

    def aggregate_atoms_(self, atom_aggregation: None | tuple[str, str] = None) -> None:
        r"""Aggregate (inplace) the atoms of the protein according to the atom_aggregation argument.

        Args:
            atom_aggregation (None | tuple[str, str]):
                The aggregation function to apply to the depth indexes of different atoms of the
                same model. The first element of the tuple is the grouping criterion. Can be one of
                the following: "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The second element of the tuple is the aggregation function to
                apply to the groups of atoms. Can be one of the following: "mean", "min", "max". The
                aggregations are applied in the order in which they appear in the sequence. Defaults
                to None.
        """

        if len(self.model.atom_types) == 0:
            return
        
        if atom_aggregation is None:
            return

        criterion: str
        function: str
        criterion, function = atom_aggregation

        if criterion not in {
            "atom_name",
            "residue_name",
            "residue_number",
            "chain_id",
            "element_symbol",
        }:
            raise ValueError(
                f"The criterion {criterion} is not valid. Valid criteria are: "
                f"atom_name, residue_name, residue_number, chain_id, element_symbol."
            )

        if function not in {"mean", "min", "max"}:
            raise ValueError(
                f"The function {function} is not valid. Valid functions are: mean, min, max."
            )

        if self.model.atom_types[0] == "X":
            return

        model = self.model.model
        if model is None:
            raise ValueError("The model is not valid.")

        mapping: dict[str | int, tuple[int]] = {
            key: tuple(model.df["ATOM"].loc[model.df["ATOM"][criterion] == key, "atom_number"])
            for key in model.df["ATOM"][criterion].unique()
        }

        aggregating_functions = {"mean": np.mean, "min": np.min, "max": np.max}
        for aggregating_function_name, aggregating_function in aggregating_functions.items():
            if aggregating_function_name == function:
                self.depth_index = np.array(
                    [
                        aggregating_function(
                            self.depth_index[np.where(np.isin(self.atom_index, mapping[key]))[0]]
                        )
                        for key in mapping.keys()
                    ]
                )

        self.atom_index = np.array(list(mapping.keys()))

    def aggregate_atoms(
        self,
        atom_aggregation: None | tuple[str, str] = None,
    ) -> SadicModelResult:
        r"""Aggregate (inplace) the atoms of the protein according to the atom_aggregation argument.

        Args:
            atom_aggregation (None | tuple[str, str]):
                The aggregation function to apply to the depth indexes of different atoms of the
                same model. The first element of the tuple is the grouping criterion. Can be one of
                the following: "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The second element of the tuple is the aggregation function to
                apply to the groups of atoms. Can be one of the following: "mean", "min", "max". The
                aggregations are applied in the order in which they appear in the sequence. Defaults
                to None.
        """
        if atom_aggregation is None:
            return self

        copy_result: SadicModelResult = deepcopy(self)
        copy_result.aggregate_atoms_(atom_aggregation)
        return copy_result


class SadicEntityResult(Repr):
    r"""Result of the SADIC algorithm for an entity of a protein.

    Can be composed of multiple models.

    Attributes:
        result_list (dict[int, SadicModelResult]):
            The list of results of the SADIC algorithm for each model of the protein.
        sadic_args (dict[str, Any]):
            The arguments of the function used to compute the SADIC depth index.
        entity (PDBEntity):
            The entity of the original protein. It represents the original pdb file with all the
            models.

    Methods:
        __init__:
            Initialize the result of the SADIC algorithm.
    """

    def __init__(
        self,
        result_list: dict[int, SadicModelResult],
        sadic_args: dict[str, Any],
        entity: PDBEntity,
    ) -> None:
        self.result_list: dict[int, SadicModelResult] = result_list
        self.sadic_args: dict[str, Any] = sadic_args
        self.entity: PDBEntity = entity

    def get_depth_index(
        self,
        filter_by: None
        | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32] = None,
        get_index: bool = False,
        model_aggregation: str | Sequence[str] = "mean",
        atom_aggregation: None | tuple[str, str] = None,
        model_aggregation_before: bool = False,
    ):
        r"""Return the depth index of the atoms of the protein.

        Can return the depth index of the atoms of the protein, filtered by a filter argument.

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
            get_index (bool):
                If True, also return the index of the atoms/residue of the protein. Defaults to
                False.
            model_aggregation (str | Sequence[str]):
                The aggregation function(s) to apply to the depth indexes of the corresponding
                atoms of different models. Can be one or more of the following: "mean", "median",
                "min", "max", "var", "std", "concatenate", "list". Defaults to "mean".
            atom_aggregation (None | tuple[str, str]):
                The aggregation function to apply to the depth indexes of different atoms of the
                same model. The first element of the tuple is the grouping criterion. Can be one of
                the following: "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The second element of the tuple is the aggregation function to
                apply to the groups of atoms. Can be one or more of the following: "mean", "min",
                "max". Defaults to None.
            model_aggregation_before (bool):
                If True, apply the model aggregation before the atom aggregation. Defaults to False.

        Returns:
            The depth index of the atoms of the protein, aggregated according to the
            model_aggregation and atom_aggregation argument.
        """
        result: SadicEntityResult
        if filter_by is not None:
            result = self.filter(filter_by=filter_by)
            return result.get_depth_index(
                get_index=get_index,
                model_aggregation=model_aggregation,
                atom_aggregation=atom_aggregation,
                model_aggregation_before=model_aggregation_before,
            )

        if not model_aggregation_before and atom_aggregation is not None:
            result = self.aggregate_atoms(atom_aggregation=atom_aggregation)
            return result.get_depth_index(
                get_index=get_index,
                model_aggregation=model_aggregation,
                model_aggregation_before=True,
            )

        output_models = {}
        output = {}

        if isinstance(model_aggregation, str):
            model_aggregation = [model_aggregation]

        model_aggregation_list = list(model_aggregation)

        if "list" in model_aggregation_list:
            output["list"] = self.aggregate_atoms(atom_aggregation).aggregate_models("list")

            if not isinstance(output["list"], dict):
                raise ValueError(
                    "The list aggregation can only be used if the atom aggregation returns a "
                    "dictionary."
                )

            if not get_index:
                output["list"] = {
                    model_index: depth_index[1]
                    for model_index, depth_index in output["list"].items()
                }

            model_aggregation_list.remove("list")

        if "concatenate" in model_aggregation_list:
            list_aggregation = (
                output["list"]
                if "list" in output
                else self.get_depth_index(get_index=True, model_aggregation="list")
            )

            if isinstance(list_aggregation, tuple):
                list_aggregation = list_aggregation[1]

            output["concatenate"] = self.aggregate_models(
                "concatenate", list_aggregation=list_aggregation
            )

            model_aggregation_list.remove("concatenate")

        if model_aggregation_list:
            concatenate_aggregation = (
                output["concatenate"]
                if "concatenate" in output
                else self.get_depth_index(get_index=True, model_aggregation="concatenate")
            )

            if not isinstance(concatenate_aggregation, tuple):
                raise ValueError(
                    "The output of the concatenate aggregation function must be a tuple"
                )

            for aggregation in model_aggregation_list:
                output_models[aggregation] = self.aggregate_models(
                    aggregation, concatenate_aggregation=concatenate_aggregation
                )

        if model_aggregation_before:
            output_model: SadicModelResult
            for output_model in output_models.values():
                output_model.aggregate_atoms_(atom_aggregation)

        for aggregation in model_aggregation_list:
            output[aggregation] = output_models[aggregation].get_depth_index(get_index=get_index)

        if len(output) == 1:
            output = list(output.values())[0]

        return output

    def save_pdb(
        self,
        path: None | str = None,
        replaced_column: None | str = "b_factor",
        model_aggregation: str = "mean",
        gzip: bool = False,
        append_newline: bool = True,
    ) -> None:
        r"""Save the entity of the protein as a PDB file.

        Args:
            path (None | str):
                The path where to save the PDB file. If None, the file is saved in the current
                directory with the name <protein_code>_sadic.pdb. Defaults to None.
            replaced_column (None | str):
                The column of the PDB file to replace with the depth index. Can be one of the
                following: "atom_number", "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol", None. If None, a new column is created. Defaults to "b_factor".
            model_aggregation (str):
                The aggregation function to apply to the depth indexes of the corresponding atoms
                of different models. Can be one of the following: "mean", "median", "min", "max",
                "var", "std". Defaults to "mean".
            gzip (bool):
                If True, the PDB file is saved in gz format. Defaults to False.
            append_newline (bool):
                If True, a newline is appended at the end of the file. Defaults to True.
        """
        if model_aggregation in {"list", "concatenate"}:
            raise ValueError(
                f"The model_aggregation argument cannot be {model_aggregation}. "
                f"Valid aggregation functions are: mean, median, min, max, var, std."
            )

        if path is None:
            path = f"./{self.entity.code}_sadic.pdb"

        if os.path.dirname(path) != '' and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        valid_columns = list(self.entity.entity.df["ATOM"].columns) + [None]
        valid_columns.remove("atom_number")
        valid_columns.remove("line_idx")
        if replaced_column not in valid_columns:
            raise ValueError(
                f"The column {replaced_column} is not valid. Valid columns are: {valid_columns}."
            )

        pdb = deepcopy(self.result_list[min(self.result_list.keys())].model.model)
        if pdb is None:
            raise ValueError("The model is not valid.")

        depth_indexes = self.get_depth_index(get_index=True, model_aggregation=model_aggregation)

        pdb.df["ATOM"][replaced_column] = np.nan

        for atom_index, depth_index in zip(depth_indexes[0].tolist(), depth_indexes[1].tolist()):
            pdb.df["ATOM"].loc[
                pdb.df["ATOM"]["atom_number"] == atom_index, replaced_column
            ] = depth_index

        pdb.to_pdb(path, gz=gzip, append_newline=append_newline)

    def save_txt(
        self, path: None | str = None, model_aggregation: str = "mean", file_format: str = "sadicv1"
    ) -> None:
        r"""Save the entity of the protein as a txt file.

        Args:
            path (None | str):
                The path where to save the txt file. If None, the file is saved in the current
                directory with the name <protein_code>_sadic.txt. Defaults to None.
            model_aggregation (str):
                The aggregation function to apply to the depth indexes of the corresponding atoms
                of different models. Can be one of the following: "mean", "median", "min", "max",
                "var", "std". Defaults to "mean".
            file_format (str):
                The format of the txt file. Can be one of the following: "sadicv1", ....
                Defaults to "sadicv1".
        """
        if model_aggregation in {"list", "concatenate"}:
            raise ValueError(
                f"The model_aggregation argument cannot be {model_aggregation}. "
                f"Valid aggregation functions are: mean, median, min, max, var, std."
            )

        if file_format not in {"sadicv1"}:
            raise ValueError(
                f"The format argument cannot be {format}. " f"Valid formats are: sadicv1."
            )

        if path is None:
            path = f"./{self.entity.code}_sadic.txt"

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        depth_indexes = self.get_depth_index(get_index=True, model_aggregation=model_aggregation)

        with open(path, "w", encoding="utf-8") as file:
            if file_format == "sadicv1":
                file.write("di\t####")
                for atom_index, depth_index in zip(depth_indexes[0], depth_indexes[1]):
                    file.write(f"\n{atom_index}\t{depth_index:.3f}")

    def summary(self) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
        r"""Return a summary of the result object.

        It includes mean and standard deviation of the depth index of the atoms of the protein,
        among the different models.
        """
        info = self.get_depth_index(get_index=True, model_aggregation=("mean", "std"))
        return info["mean"][0], info["mean"][1], info["std"][1]

    def filter_(
        self,
        filter_by: None
        | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32] = None,
    ) -> None:
        r"""Filter (inplace) the entity inside the result object.

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
        if filter_by is not None:
            self.entity.filter_(filter_by)

            for model_result in self.result_list.values():
                model_result.filter_(filter_by)

    def filter(
        self,
        filter_by: None
        | dict[str, str | int | Sequence[str] | Sequence[int]]
        | tuple[NDArray[np.float32], float]
        | NDArray[np.float32] = None,
    ) -> SadicEntityResult:
        r"""Filter the entity inside the result object according to the filter_by argument.

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

        Returns (SadicEntityResult):
            The filtered result object.
        """
        if filter_by is None:
            return self

        result_copy: SadicEntityResult = deepcopy(self)
        result_copy.filter_(filter_by)
        return result_copy

    def aggregate_atoms_(self, atom_aggregation: None | tuple[str, str] = None) -> None:
        r"""Aggregate (inplace) the atoms of the entity according to the atom_aggregation argument.

        Args:
            atom_aggregation (None | tuple[str, str]):
                The aggregation function to apply to the depth indexes of different atoms of the
                same model. The first element of the tuple is the grouping criterion. Can be one of
                the following: "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The second element of the tuple is the aggregation function to
                apply to the groups of atoms. Can be one or more of the following: "mean", "min",
                "max". Defaults to None.
        """
        if atom_aggregation is None:
            return

        model_result: SadicModelResult
        for model_result in self.result_list.values():
            model_result.aggregate_atoms_(atom_aggregation)

    def aggregate_atoms(
        self,
        atom_aggregation: None | tuple[str, str] = None,
    ) -> SadicEntityResult:
        r"""Aggregate the atoms of the entity according to the atom_aggregation argument.

        Args:
            atom_aggregation (None | tuple[str, str]):
                The aggregation function to apply to the depth indexes of different atoms of the
                same model. The first element of the tuple is the grouping criterion. Can be one of
                the following: "atom_name", "residue_name", "residue_number", "chain_id",
                "element_symbol". The second element of the tuple is the aggregation function to
                apply to the groups of atoms. Can be one or more of the following: "mean", "min",
                "max". Defaults to None.

        Returns:
            SadicEntityResult (SadicEntityResult):
                A copy of the current object with the atoms aggregated.
        """
        if atom_aggregation is None:
            return self

        copy_result: SadicEntityResult = deepcopy(self)
        copy_result.aggregate_atoms_(atom_aggregation)
        return copy_result

    def aggregate_models(
        self,
        model_aggregation: str = "mean",
        list_aggregation: None | dict = None,
        concatenate_aggregation=None,
    ):
        r"""Aggregate the models of the entity and return a single summary model result.

        Args:
            model_aggregation (str):
                The aggregation function to apply to the depth indexes of the corresponding atoms
                of different models. Can be one of the following: "mean", "median", "min", "max",
                "var", "std", "concatenate", "list". Defaults to "mean".
            list_aggregation (None | dict):
                The result of the list aggregation function. Should be present when using the
                concatenate aggregation function. Defaults to None.
            concatenate_aggregation (None | dict):
                The result of the concatenate aggregation function. Should be present when using an
                aggregation function other than list or concatenate. Defaults to None.

        Returns:
            The depth index of the atoms of the protein, aggregated according to the
            model_aggregation and atom_aggregation argument.
        """
        if model_aggregation not in {
            "mean",
            "median",
            "min",
            "max",
            "var",
            "std",
            "list",
            "concatenate",
        }:
            raise ValueError(
                f"Invalid aggregation function {model_aggregation}. Must be one of the following: "
                f"mean, median, min, max, var, std, concatenate, list."
            )

        if model_aggregation == "list":
            list_output: dict[int, tuple[NDArray[np.int32], NDArray[np.float32]]] = {}

            model_index: int
            model_result: SadicModelResult
            for model_index, model_result in self.result_list.items():
                list_output[model_index] = model_result.get_depth_index(get_index=True)

            return list_output

        if model_aggregation == "concatenate":
            if list_aggregation is None:
                raise ValueError(
                    "The list_aggregation argument must be specified when using the "
                    "concatenate aggregation function."
                )

            atom_indexes: NDArray[np.int32] = np.sort(
                np.unique(np.concatenate([indexes for indexes, _ in list_aggregation.values()]))
            )

            atom2array_indexes: dict[int, int] = {
                atom_index: array_index for array_index, atom_index in enumerate(atom_indexes)
            }

            model_indexes = np.array(list(list_aggregation.keys()), dtype=np.int32)

            concatenate_output: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]

            concatenate_output = (
                model_indexes,
                atom_indexes,
                np.full((len(list_aggregation), len(atom_indexes)), np.nan, dtype=np.float32),
            )

            model2array_indexes: dict[int, int] = {
                model_index: array_index for array_index, model_index in enumerate(list_aggregation)
            }

            for model_index, (indexes, depth_indexex) in list_aggregation.items():
                for index, depth_index in zip(indexes, depth_indexex):
                    concatenate_output[2][
                        model2array_indexes[model_index], atom2array_indexes[index]
                    ] = depth_index

            return concatenate_output

        if concatenate_aggregation is None:
            raise ValueError(
                "The concatenate_aggregation argument must be specified when using an aggregation "
                "function other than list or concatenate."
            )

        aggregating_functions = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "min": np.nanmin,
            "max": np.nanmax,
            "var": np.nanvar,
            "std": np.nanstd,
        }

        model: Model = self.result_list[min(self.result_list.keys())].model
        if model.model is None:
            raise ValueError("The model is not valid.")

        atoms = []
        for model_result in self.result_list.values():
            current_model = model_result.model.model
            if current_model is None:
                raise ValueError("The model is not valid.")
            atoms.append(current_model.df["ATOM"])

        model.model.df["ATOM"] = (
            pd.concat(atoms).drop_duplicates("atom_number").sort_values("atom_number")
        )

        output_model_result: SadicModelResult = SadicModelResult(
            concatenate_aggregation[1],
            aggregating_functions[model_aggregation](concatenate_aggregation[2], axis=0),
            model,
        )

        return output_model_result
