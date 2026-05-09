import numpy as np
import numpy.typing as npt
from typing import Any

def manhattan_norm(x1: npt.NDArray[np.float64],
                   x2: npt.NDArray[np.float64],
                   axis: int) -> Any:
    return np.linalg.norm(x1 - x2, axis= axis, ord= 1)

def euclidean_norm(x1: npt.NDArray[np.float64],
                   x2: npt.NDArray[np.float64],
                   axis: int) -> Any:
    return np.linalg.norm(x1 - x2, axis= axis, ord= 2)

def minkowski(x1: npt.NDArray[np.float64], 
              x2: npt.NDArray[np.float64],
              axis: int,
              ord: int) -> Any:
    return np.linalg.norm(x1 - x2, axis= axis, ord= ord)