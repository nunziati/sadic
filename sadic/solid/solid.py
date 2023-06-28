r"""Defines the Solid abstract class."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class Solid(ABC):
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
