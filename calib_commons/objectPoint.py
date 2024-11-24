import numpy as np
from dataclasses import dataclass

from calib_commons.types import idtype

@dataclass
class ObjectPoint: 
    id: idtype
    position: np.ndarray
    color: str = None
        
       