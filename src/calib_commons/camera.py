import numpy as np 

from calibCommons.viz.visualization_tools import plot_camera
from calibCommons.intrinsics import Intrinsics
from calibCommons.types import idtype
from calibCommons.utils.se3 import SE3

from calibCommons.utils import utils

class Camera: 

    def __init__(self,
                 id: idtype,
                 pose: SE3, 
                 intrinsics: Intrinsics):
        
        self.id = id
        self.pose = pose
        self.intrinsics = intrinsics
        
    
    def get_projection_matrix(self) -> np.ndarray: 
        return self.intrinsics.compute_projection_matrix(self.pose.mat)
    
    def reproject(self, 
                  object_points_in_world) -> np.ndarray: 
        return utils.reproject(self.intrinsics.compute_projection_matrix(self.pose.mat), object_points_in_world)
        
    def plot(self, name = None, ax=None) -> None: 
        if name == None:
            name = r"$\{C_{" + str(self.id) + r"}\}$"
        plot_camera(self.pose.mat, name, size=0.3, ax=ax)
       

    def backproject_points_using_plane(self, 
                                       pi: np.ndarray, 
                                       image_pts: np.ndarray): 
        if image_pts.ndim == 1 and image_pts.shape[0] == 2: 
            image_pts = image_pts[np.newaxis,:]
        P = self.get_projection_matrix()
        X = np.linalg.inv(np.vstack((P, pi))) @ np.vstack((image_pts.T, np.vstack((  np.ones(image_pts.shape[0]), np.zeros(image_pts.shape[0]) )) ))
        X = X / X[3,:]
        X = X[:3,:].T
        return X

       