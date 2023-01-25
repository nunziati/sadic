from abc import ABC, abstractclassmethod

class Quantizer(ABC):
    def __init__(self):
        pass

class CartesianQuantizer(Quantizer):
    def __init__(self):
        super().__init__()

class SphericalQuantizer(Quantizer):
    def __init__(self):
        super().__init__()