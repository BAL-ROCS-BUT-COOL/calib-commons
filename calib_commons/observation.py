import numpy as np


class Observation: 

    def __init__(self, 
                 _2d: np.ndarray = None, 
                 is_conform: bool = True):
        self._2d = _2d 
        self._is_conform = is_conform


    