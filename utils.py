from mytyping import PointType
from mytyping import is_PointType

def point_square_distance(x, y) -> float:
    if not is_PointType(x) or not is_PointType(y):
        raise TypeError("x and y must be PointType")
    
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2