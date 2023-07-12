r"""init file for pdb package."""
from .pdbentity import Model, PDBEntity
from .result import SadicModelResult, SadicEntityResult

__all__ = ["Model", "PDBEntity", "SadicModelResult", "SadicEntityResult"]
