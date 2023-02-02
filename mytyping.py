from collections.abc import Sequence
from numbers import Number

from numpy.typing import NDArray
import numpy as np

NumpyNumberType = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64 | np.float16 | np.float32 | np.float64 | np.float128
NumberType = NumpyNumberType | float | int
PointType = Sequence[NumberType] | NDArray[NumpyNumberType]
PointSequenceType = Sequence[PointType] | NDArray[NumpyNumberType]
NumberSequenceType = Sequence[NumberType] | NDArray[NumpyNumberType]

def is_PointType(obj):
    return (
        isinstance(obj, Sequence) and len(obj) == 3 and all(isinstance(o, Number) for o in obj) or
        isinstance(obj, np.ndarray) and obj.ndim == 1 and obj.shape == (3,) and all(isinstance(o, Number) for o in obj)
    )

def is_PointSequenceType(obj):
    return (
        isinstance(obj, Sequence) and len(obj) > 0 and all(is_PointType(o) for o in obj) or
        isinstance(obj, np.ndarray) and obj.ndim == 2 and all(is_PointType(o) for o in obj)
    )

def is_NumberSequenceType(obj):
    return (
        isinstance(obj, Sequence) and len(obj) > 0 and all(isinstance(o, Number) for o in obj) or
        isinstance(obj, np.ndarray) and obj.ndim == 1 and all(isinstance(o, Number) for o in obj)
    )