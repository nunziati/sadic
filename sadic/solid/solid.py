from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class Solid(ABC):
    @abstractmethod
    def __init__(self, *args) -> None:
        pass
    
    @abstractmethod
    def is_inside(self, *args) -> NDArray[np.bool_]:
        pass

    @abstractmethod
    def get_extreme_coordinates(self, *args) -> NDArray[np.float32]:
        pass