import numpy as np
from typing import Tuple

# from externalCalibrationPlanarPoints.utils import utils
from calibCommons.utils import utils
class Intrinsics: 

    def __init__(self, 
                 K: np.ndarray, 
                 resolution: Tuple = None): 
        self.K = K
        self.resolution = resolution

    def valid_points(self, points) -> np.ndarray:
        """
        Determines the indices of valid points that are within the image boundaries.
        Args:
            points (numpy.ndarray): A 2xN matrix of points, where each row represents x or y coordinates.
        Returns:
            numpy.ndarray: A boolean array where True indicates a valid point.
        """
        
        assert points.shape[1] == 2, 'Input must be a Nx2 matrix of points'

        x = points[:, 0]
        y = points[:, 1]
        valid_x = (x > 0) & (x < self.resolution[0])
        valid_y = (y > 0) & (y < self.resolution[1])
        valid_idx = valid_x & valid_y  
        return valid_idx

    def compute_projection_matrix(self, pose) -> np.ndarray: 
        R_W__C = pose[0:3, 0:3]
        C_W = pose[0:3, 3]
        R = R_W__C.T
        t = -R @ C_W
        Rt = np.hstack((R, t.reshape(-1, 1)))
        P = self.K @ Rt
        return P
    
    def reproject(self, pose, _3d): 
        return utils.reproject(self.compute_projection_matrix(pose), _3d)
    
    def get_normalized_points(self, points: np.ndarray): 
        points_hom = np.hstack((points, np.ones((points.shape[0],1))))
        points_normalized_hom = np.linalg.inv(self.K) @ points_hom.T
        return points_normalized_hom[:2,:].T



