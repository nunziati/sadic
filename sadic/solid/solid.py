from abc import ABC, abstractmethod

class Solid(ABC):
    @abstractmethod
    def __init__(self, *args):
        pass
    
    @abstractmethod
    def is_inside(self, *args):
        raise NotImplementedError

    @abstractmethod
    def get_extreme_coordinates(self, *args):
        raise NotImplementedError