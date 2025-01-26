from typing import Tuple

import numpy as np
import scipy.spatial.transform
import cv2

def change_coords(T_2_1, P_1): 
    if P_1.ndim == 1 and P_1.shape[0] == 3:
        P_2_bar = T_2_1 @ np.append(P_1, 1)
        P_2 = P_2_bar[:3]
        return P_2
    P_2_bar = T_2_1 @ np.vstack((P_1.T, np.ones(P_1.shape[0])))
    P_2 = P_2_bar[:3,:].T
    return P_2

def inv_T(T):
    R_inv = T[:3, :3].T
    t = T[:3, 3]

    T_inv = np.vstack((np.hstack((R_inv, -R_inv @ t[:, None])), [0, 0, 0, 1]))
    return T_inv

def T_from_rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    return T

def rvec_tvec_from_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    rvec = scipy.spatial.transform.Rotation.from_matrix(T[:3, :3]).as_rotvec()
    tvec = T[:3, 3]
    # return rvec[:, np.newaxis], tvec[:, np.newaxis]
    return rvec, tvec

def T_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray: 
    R, _ = cv2.Rodrigues(rvec)
    return T_from_rt(R, tvec)

def T_from_q(q: np.ndarray) -> np.ndarray:
    return T_from_rt(scipy.spatial.transform.Rotation.from_euler('ZYX', q[:3], degrees=False).as_matrix(), q[3:])

def q_from_T(T: np.ndarray) -> np.ndarray: 
    eul = scipy.spatial.transform.Rotation.from_matrix(T[:3, :3]).as_euler('ZYX', degrees=False)
    t = T[:3, 3]
    q = np.zeros(6)
    q[:3] = eul
    q[3:] = t
    return q

def T_from_quat_t(quat: np.ndarray, t: np.ndarray, scalar_first=False) -> np.ndarray:
    return T_from_rt(scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=scalar_first).as_matrix(), t)


class SE3:
    def __init__(self, mat):
        if not self.is_valid_se3(mat):
            raise ValueError("The matrix is not a valid SE(3) transformation matrix")
        self.mat = mat
    
    @staticmethod
    def is_valid_se3(mat):
        # Check if the matrix is 4x4
        if mat.shape != (4, 4):
            return False
        
        # Extract the rotation part and the translation part
        R = mat[:3, :3]
        t = mat[:3, 3]
        bottom_row = mat[3, :]
        
        # Check if the bottom row is [0, 0, 0, 1]
        if not np.allclose(bottom_row, [0, 0, 0, 1]):
            return False
        
        # # Check if R is a valid rotation matrix: R^T R = I and det(R) = 1
        # if not np.allclose(np.dot(R.T, R), np.eye(3)):
        #     return False
        
        if not np.isclose(np.linalg.det(R), 1):
            return False
        
        return True
    
    def __repr__(self):
        return f"SE3(T=\n{self.mat})"
    
    def inv(self):
        return SE3(inv_T(self.mat))

    def __mul__(self, other):
        if isinstance(other, SE3):
            return SE3(np.dot(self.mat, other.mat))
        else:
            raise ValueError("Multiplication is only supported with another SE3 object")
    
    def transform_point(self, point):
        """Transforms a 3D point using the SE3 transformation."""
        point_homogeneous = np.append(point, 1)
        transformed_point = np.dot(self.mat, point_homogeneous)
        return transformed_point[:3]

    def get_R(self):
        """Returns the rotation matrix part of the SE3 transformation."""
        return self.mat[:3, :3]
    
    def get_t(self):
        """Returns the translation vector part of the SE3 transformation."""
        return self.mat[:3, 3]
    
    def get_x(self):
        """Returns the x component of the translation vector."""
        return self.mat[0, 3]
    
    def get_y(self):
        """Returns the y component of the translation vector."""
        return self.mat[1, 3]
    
    def get_z(self):
        """Returns the z component of the translation vector."""
        return self.mat[2, 3]

    # class methods
    
    @classmethod
    def from_rt(cls, R, t):
        """Creates an SE3 object from a rotation matrix R and translation vector t."""
        return cls(T_from_rt(R, t))

    @classmethod
    def from_q(cls, q):
        return cls(T_from_q(q))
    
    @classmethod
    def from_rvec_tvec(cls, rvec, tvec):
        return cls(T_from_rvec_tvec(rvec, tvec))   

    # @classmethod
    # def inv(cls, mat):
    #     return cls(inv_T(mat))
    
    @classmethod
    def idendity(cls):
        return cls(np.eye(4))