r"""Defines the Solid abstract class."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from sadic.utils import Repr


class Solid(ABC, Repr):
    r"""Abstract class for solids

    Attributes:
        None

    Methods:
        __init__:
            Abstract constructor for solids.
        get_extreme_coordinates:
            Abstract method to get the extreme coordinates of the solid.
    """

    @abstractmethod
    def __init__(self) -> None:
        r"""Abstract constructor for solids."""

    @abstractmethod
    def get_extreme_coordinates(self) -> NDArray[np.float32]:
        r"""Abstract method to get the extreme coordinates of the solid.

        Returns:
            NDArray[np.float32]:
                A numpy array of floats representing the extreme coordinates of the solid.
        """

    @abstractmethod
    def is_inside(self, *args, **kwargs) -> NDArray[np.bool_]:
        r"""Checks if a (set of) point(s) is inside the solid.

        Particular quantizers may implement this method for other structures than points.
        """
