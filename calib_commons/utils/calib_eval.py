import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation 
from pathlib import Path
from typing import Dict, Any
import itertools
import matplotlib.pyplot as plt

from calib_commons.data.data_json import load_cameras_poses_from_json
from calib_commons.correspondences import get_conform_obs_of_cam, filter_correspondences_with_track_length, get_tracks
from calib_commons.types import idtype

from calib_commons.observation import Observation
from calib_commons.data.load_calib import load_intrinsics
from calib_commons.intrinsics import Intrinsics
from calib_commons.data.data_pickle import load_from_pickle
from calib_commons.camera import Camera
from calib_commons.utils.se3 import SE3
from calib_commons.scene import Scene, SceneType
from calib_commons.objectPoint import ObjectPoint

from calib_commons.utils.calib_eval_config import CalibEvaluatorConfig
import copy 
# from convert import convert_correspondences

from calib_commons.viz.visualization_tools import get_color_from_id

class CalibEvaluator:
    def __init__(self, correspondences: Dict[Any, Dict[Any, Observation]], 
                 intrinsics: Dict[Any, np.ndarray], 
                 camera_poses: Dict[Any, Dict[str, Any]], 
                 config: CalibEvaluatorConfig):
        """
        Initializes the EvalCalib class.
        """
        self.correspondences = copy.deepcopy(correspondences)
        cameras: Dict[str, Camera] = {}
        
        # Sort cameras by their IDs (assuming IDs are in the format 'gopro1', 'gopro2', ..., 'gopro10')
        # sorted_camera_ids = sorted(camera_poses.keys(), key=lambda x: int(x.replace('gopro', '')))
        
        for cam_id in camera_poses:
            euler_angles = camera_poses[cam_id]["euler_ZYX"]
            R = Rotation.from_euler('ZYX', euler_angles, degrees=True).as_matrix()
            t = np.array(camera_poses[cam_id]["t"]).reshape(3, 1)
            cameras[cam_id] = Camera(cam_id, SE3.from_rt(R,t), intrinsics[cam_id])
        self.scene = Scene(scene_type=SceneType.ESTIMATE, cameras = cameras) 

        self.config = config

    def set_cameras(self, camera_poses): 
        for cam_id in camera_poses:
            euler_angles = camera_poses[cam_id]["euler_ZYX"]
            R = Rotation.from_euler('ZYX', euler_angles, degrees=True).as_matrix()
            t = np.array(camera_poses[cam_id]["t"]).reshape(3, 1)
            self.scene.cameras[cam_id].pose = SE3.from_rt(R,t)

    def compute_initial_estimates(self):
        """
        Computes an initial estimate of each 3D point using pairwise triangulation and median filtering.

        :return: Dictionary of initial 3D point estimates.
        """
        # Reorganize correspondences to group by points instead of cameras
        point_correspondences = self._group_by_point_id()

        for pt_id, cam_points in point_correspondences.items():
            # Collect all camera pairs for this 3D point
            camera_ids = [cam_id for cam_id, obs in cam_points.items() if obs._is_conform]
            if len(camera_ids) < 2:
                continue  # Need at least 2 views for triangulation

            # Compute pairwise 3D points for all camera pairs
            pairwise_3d_points = []
            
            combinations = list(itertools.combinations(camera_ids, 2))
            for cam1_id, cam2_id in combinations:
                P1, P2 = self.scene.cameras[cam1_id].get_projection_matrix(), self.scene.cameras[cam2_id].get_projection_matrix()
                pt1, pt2 = self.correspondences[cam1_id][pt_id]._2d, self.correspondences[cam2_id][pt_id]._2d
                
                # Triangulate the point between the two cameras
                pts4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1)).T
                pts3d = pts4d[:, :3] / pts4d[:, 3]  # Convert from homogeneous to Euclidean coordinates
                pairwise_3d_points.append(pts3d[0])

            # Use the median of the pairwise estimates as the initial guess
            initial_3d_point = np.mean(pairwise_3d_points, axis=0)
            self.scene.add_point(ObjectPoint(id=pt_id, position=initial_3d_point))
            # self.scene.add_point(ObjectPoint(id=pt_id, position=np.array([0,0,0])))

        return 

    def iterative_filtering(self): 
        print(" ")
        print("----- Iterative Filtering -----")
        iter = 1
        continue_ = True
        while continue_:
            print(" ")
            print(f"----- Iteration: {iter} -----" )
            iter += 1
            print("** BA: **")
            self.refine_points()
            print(" ")
            print("** Filtering: **")

            # if self.config.display_reprojection_errors:
            #     self.display_histogram_reprojection_errors()
            continue_ = self.filtering() 
    
    def filtering(self) -> bool: 
        num_point_filtered = 0
        points_removed = []

        point_ids = sorted(list(self.scene.object_points.keys()))
        for camera in self.scene.cameras.values(): 
            # print("")
            # print(f"camera {camera.id}")
            for point_id in point_ids: 
                # print(f"point estimate {point_id}")
                point = self.scene.object_points.get(point_id)
                if point:
                    # print(point)
                    observation = self.correspondences[camera.id].get(point.id)
                    if observation and observation._is_conform:
                        _2d_reprojection = camera.reproject(point.position)
                        errors_xy = observation._2d - _2d_reprojection
                        error = np.sqrt(np.sum(errors_xy**2))
                        # print()
                        is_obsv_conform = error < self.config.reprojection_error_threshold
                        # if self.config.display_reprojection_errors:
                        # print(f"observation of point {point.id:>3} in cam {camera.id} error: {error:.2f} [pix]")
                        if not is_obsv_conform: 
                            print(f"observation of point {point.id:>3} in cam {camera.id} error: {error:.2f} [pix]")

                            num_point_filtered += 1
                            self.correspondences[camera.id][point.id]._is_conform = False 

                            
                            if len(self.get_tracks()[point.id]) < self.config.min_track_length:

                                print(f"removing point {point.id} from estimate")
                                points_removed.append(point.id)
                                # self.p.append(checker.id)   
                                del self.scene.object_points[point.id]
                                # if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
                                #     del self.estimate._2d_plane_points[point.id]


        if self.config.display: 
            print(f" -> Number of observations filtered: {num_point_filtered}")
            print(f" -> Points removed from estimate: {sorted(points_removed)}")
        return num_point_filtered > 0

    def refine_points(self):
        """
        Refines each 3D point independently by minimizing its reprojection error in all cameras.

        :return: Dictionary of refined 3D points.
        """
        for pt_id, point in self.scene.object_points.items():
            # Gather corresponding 2D points and camera matrices for this point
            cam_matrices = []
            points_2d = []
            x = point.position

            # Collect observations from all cameras for this point
            for cam_id in self.correspondences:
                if pt_id in self.correspondences[cam_id] and self.correspondences[cam_id][pt_id]._is_conform:
                    cam_matrices.append(self.scene.cameras[cam_id].get_projection_matrix())
                    points_2d.append(self.correspondences[cam_id][pt_id]._2d)

            # Optimize using reprojection error minimization
            result = least_squares(
                self._reprojection_error,
                x,
                method="lm",
                ftol=self.config.ba_least_square_ftol,
                x_scale="jac",
                args=(cam_matrices, points_2d)
            )
            self.scene.object_points[pt_id].position = result.x

        return

    def _reprojection_error(self, point_3d, camera_matrices, points_2d):
        """
        Computes the reprojection error for a given 3D point across multiple cameras.

        :param point_3d: 3D point to be refined.
        :param camera_matrices: List of camera projection matrices.
        :param points_2d: List of corresponding 2D points in each camera.
        :return: Flattened array of reprojection errors.
        """
        point_3d_hom = np.hstack([point_3d, 1.0])  # Convert to homogeneous coordinates
        residuals = []

        for P, pt_2d in zip(camera_matrices, points_2d):
            projected_point = P @ point_3d_hom.T
            projected_point /= projected_point[2]  # Normalize by the third row
            reprojection = projected_point[:2]
            residuals.extend((reprojection - pt_2d).ravel())

        return np.array(residuals)

    def _group_by_point_id(self):
        """
        Converts the camera-centric correspondences to point-centric correspondences.

        :return: Dictionary with point IDs as keys and camera-point observations as values.
        """
        point_correspondences = {}
        for cam_id, points in self.correspondences.items():
            for pt_id, obs in points.items():
                if pt_id not in point_correspondences:
                    point_correspondences[pt_id] = {}
                point_correspondences[pt_id][cam_id] = obs

        return point_correspondences


    def get_tracks(self) -> Dict[idtype, set[idtype]]: 
       return get_tracks(self.correspondences)

    def get_scene(self) -> Scene:
       
        # elif world_frame == WorldFrame.CAM_ID_1: 
        #     T = se3.inv_T(self.estimate.scene.cameras[1].pose.mat)
        # else: 
        #     raise ValueError("World Frame not valid.")
        
        # cameras = {id: Camera(id, SE3(T @ camera.pose.mat), camera.intrinsics) for id, camera in self.estimate.scene.cameras.items()}
        # object_points = {id: ObjectPoint(id, (T @ np.append(p.position, 1))[:3]) for id, p in self.estimate.scene.object_points.items()}
        # if self.estimate.plane: 
        #     plane = Plane(pose=SE3(T@self.estimate.plane.pose.mat))
        # else: 
        #     plane = None


        scene =  Scene(cameras=self.scene.cameras, 
                      object_points=self.scene.object_points, 
                      scene_type=SceneType.ESTIMATE)
        
        for point_id in self.scene.object_points: 
            checker_id, point_index = point_id.split('_')
            scene.object_points[point_id].color = get_color_from_id(checker_id)

        return scene