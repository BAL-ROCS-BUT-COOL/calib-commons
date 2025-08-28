import json
import os
import cv2
import numpy as np
from typing import Optional

from calib_commons.intrinsics import Intrinsics


def _extract_opencv_intrinsics(block: dict):
    """Helper to read OpenCV-style intrinsics/distortion from a JSON block."""
    K = np.array(block['intrinsics']['data']).reshape(3, 3).astype(np.float32)
    dist = np.array(block['distortionCoefficients']['data']).astype(np.float32)
    return K, dist


def load_intrinsics(path: str, cam: Optional[str] = None):
    """
    Load intrinsics either from a single-camera JSON (cam=None)
    or from a joint JSON that contains multiple cameras (cam='<name>').
    """
    with open(path, "r") as f:
        data = json.load(f)

    if cam is None:
        # single-file schema
        return _extract_opencv_intrinsics(data['sensors']['RGB'])
    else:
        # joint-file schema keyed by camera name
        if cam not in data:
            raise KeyError(f"Camera '{cam}' not found in joint intrinsics \
                           file: {path}")
        return _extract_opencv_intrinsics(data[cam]['sensors']['RGB'])


def construct_cameras_intrinsics(
        images_parent_folder,
        intrinsics_path_or_dir
) -> Intrinsics:
    """
    Build a dict of Intrinsics objects for all camera subfolders under
    images_parent_folder.
    intrinsics_path_or_dir can be either:
      - a directory containing <cam>_intrinsics.json files, or
      - a single joint JSON file with all cameras.
    """
    image_folders = {
        cam: os.path.join(images_parent_folder, cam)
        for cam in os.listdir(images_parent_folder)
        if os.path.isdir(os.path.join(images_parent_folder, cam))
    }

    use_dir = os.path.isdir(intrinsics_path_or_dir)
    if use_dir:
        intrinsics_paths = {
            cam: os.path.join(intrinsics_path_or_dir, cam + "_intrinsics.json")
            for cam in image_folders
        }

    intrinsics = {}
    for cam in image_folders:
        if use_dir:
            intrinsics[cam] = construct_camera_intrinsics(
                image_folders[cam], intrinsics_paths[cam], cam=None
            )
        else:
            # joint file: pass the same file path but specify cam name
            intrinsics[cam] = construct_camera_intrinsics(
                image_folders[cam], intrinsics_path_or_dir, cam=cam
            )
    return intrinsics


def construct_camera_intrinsics(
    images_folder: str,
    intrinsics_path: str,
    cam: Optional[str] = None
) -> Intrinsics:

    # Find all images in the folder
    image_files = [f for f in os.listdir(images_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Take the image with the smallest number
    first_image_path = os.path.join(images_folder, image_files[0])

    # Load the image
    image = cv2.imread(first_image_path)
    resolution = (image.shape[1], image.shape[0])

    K, _ = load_intrinsics(intrinsics_path, cam=cam)
    return Intrinsics(K, resolution)
