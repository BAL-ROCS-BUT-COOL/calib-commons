import numpy as np
import cv2 
from skspatial.objects import Points, Plane
from typing import Tuple

# from calibCommons.utils


def view_score(image_points: np.ndarray, 
                   image_resolution: Tuple):
        s = 0
        L = 3
        width, height = image_resolution
        for l in range(1, L+1):
            K_l = 2**l
            w_l = K_l  # Assuming w_l = K_l is correct
            grid = np.zeros((K_l, K_l), dtype=bool)
            for point in image_points:
                u,v = point
                u = np.clip(u, 0, width-1)
                v = np.clip(v, 0, height-1)
                
                x = int(np.ceil(K_l * u / width)) - 1
                y = int(np.ceil(K_l * v / height)) - 1
              
                if grid[x, y] == False:
                    grid[x, y] = True  # Mark the cell as full
                    s += w_l  # Increase the score
        return s

def reproject(P: np.ndarray, 
                objectPointsinWorld) -> np.ndarray: 
    assert (objectPointsinWorld.ndim == 2 and objectPointsinWorld.shape[1] == 3) or (objectPointsinWorld.ndim == 1 and objectPointsinWorld.shape[0] == 3), "Array must have 3 columns"

    if (objectPointsinWorld.ndim == 1 and objectPointsinWorld.shape[0] == 3): 
         objectPointsinWorld = objectPointsinWorld[:,None].T
    augmentedPoints = np.ones((4, objectPointsinWorld.shape[0]))
    augmentedPoints[:3,:] = objectPointsinWorld.T

    # _2dHom = P @ np.vstack((objectPointsinWorld.T, np.ones((1, objectPointsinWorld.shape[0]))))
    _2dHom = P @ augmentedPoints
    _2d = _2dHom / _2dHom[2,:]
    _2d = _2d[:2, :].T

    return _2d 



