import numpy as np
import cv2 
from typing import Tuple

# from calib_commons.utils


def K_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0,  cx], 
                     [0,  fy, cy], 
                     [0,  0,  1]])

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


def criticality_score(boards: list[np.ndarray], img_shape: tuple[int, int], granularity=100) -> list[int]:
    
    shape = (img_shape[1] // granularity + 1, img_shape[0] // granularity + 1)
    grid = np.zeros(shape)

    # Insert points into bins
    for board in boards:
         for u, v in board:
              x, y = int(u // granularity), int(v // granularity)
              grid[x, y] += 1

    scores = []
    for board in boards:
        score = 0
        for u, v in board:
            x, y = int(u // granularity), int(v // granularity)
            score += 1. / grid[x, y]
        scores.append(score)
    
    return scores


def discard_blurry(paths, percentile=25):
    blur = []
    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Exclude images with too much blur
        blur_measure = cv2.Laplacian(gray, ddepth=cv2.CV_64F).var()
        blur.append(blur_measure)

    threshold = np.percentile(blur, percentile)
    indices = np.where(blur <= threshold)[0]

    return [p for i, p in enumerate(paths) if i not in indices]

def blur_score(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return cv2.Laplacian(gray, ddepth=cv2.CV_64F).var()

