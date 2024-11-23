from typing import List, Dict
import numpy as np
from enum import Enum 
from calibCommons.camera import Camera
from calibCommons.objectPoint import ObjectPoint
from calibCommons.observation import Observation
from calibCommons.intrinsics import Intrinsics
from calibCommons.types import idtype
from calibCommons.utils.se3 import q_from_T

from calibCommons.data.data_json import save_cameras_poses_to_json

class SceneType(Enum): 
    SYNTHETIC = "Synthetic"
    ESTIMATE = "Estimate"

class Scene: 

    def __init__(self, 
                 cameras: Dict[idtype, Camera] = None, 
                 object_points: Dict[idtype, ObjectPoint] = None, 
                 scene_type: SceneType = None): 
        
        if cameras is None:
            self.cameras = {}
        else:
            self.cameras = cameras

        if object_points is None:
            self.object_points = {}
        else:
            self.object_points = object_points

        self.type = scene_type
        self.reprojections = None

        if self.type is None: 
            raise ValueError("Scene type must be provided.")
        
        if self.type == SceneType.ESTIMATE:
            self._generate_reprojections(0, False)

    def generate_noisy_observations(self, noise_std) -> None:
        if self.type == SceneType.ESTIMATE:
            raise ValueError("Not recommended to generate noisy observations on a scene estimate. You may instead want to generate exact reprojections.") 
    
        if self.reprojections is not None: 
            raise ValueError("Noisy observations have already been generated.") 
        
        self._generate_reprojections(noise_std=noise_std, mask_outside_fov=True)

    def _generate_reprojections(self, noise_std, mask_outside_fov) -> None: 
        correspondences = {} # dict by cam id

        for camera in self.cameras.values(): 
            correspondences[camera.id] = {}
            for object_point in self.object_points.values(): 
                _2d = camera.reproject(object_point.position)
                _2d += noise_std * np.random.normal(size=(_2d.shape[0], 2))
                within_fov = all(camera.intrinsics.valid_points(_2d))
                if not mask_outside_fov or within_fov:
                    correspondences[camera.id][object_point.id] = Observation(_2d.squeeze())

        self.reprojections = correspondences

    def get_intrinsics(self) -> Dict[idtype, Intrinsics]: 
        return {camera.id: camera.intrinsics for camera in self.cameras.values()}

    def get_correspondences(self): 
        return self.reprojections
    
    def add_camera(self, camera: Camera) -> None: 
        if self.cameras is None: 
            self.cameras = {}
        self.cameras[camera.id] = camera

    def add_point(self, point: ObjectPoint) -> None: 
        if self.object_points is None: 
            self.object_points = {}
        self.object_points[point.id] = point

    def get_camera_ids(self) -> set[idtype]: 
        return set(self.cameras.keys())

    def get_num_cameras(self) -> int: 
        return len(self.cameras)
    
    def get_point_ids(self) -> set[idtype]: 
        return set(self.object_points.keys())

    def get_num_points(self) -> int: 
        return len(self.object_points)

    def print_cameras_poses(self): 
        for camera in self.cameras.values(): 
            q = q_from_T(camera.pose.mat)
            euler = q[:3] * 180 / np.pi
            t = q[3:]
            print(f"cam {camera.id}: euler = {euler}, t = {t}")

    def save_cameras_poses_to_json(self, path): 
        save_cameras_poses_to_json(self.cameras, path)  
