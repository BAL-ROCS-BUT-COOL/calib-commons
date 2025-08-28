import os
from enum import Enum
from typing import Dict, Optional

import cv2 as cv
import numpy as np
from tqdm import tqdm

from calib_commons.data.load_calib import load_intrinsics


class BoardType(Enum):
    CHESSBOARD = "chessboard"
    CHARUCO = "charuco"


def detect_board_corners(
    images_parent_folder: str,
    board_type: BoardType,
    charuco_detector: cv.aruco.CharucoDetector,
    columns: int,
    rows: int,
    intrinsics_path: str,
    undistort: bool = True,
    display: bool = False,
    save_images_with_overlayed_detected_corners: bool = True,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Dispatch to the appropriate detector and return a nested dict of
    correspondences. intrinsics_path can be a directory (per-camera JSONs)
    or a joint JSON file.
    """
    if board_type == BoardType.CHESSBOARD:
        return detect_chessboards(
            images_parent_folder=images_parent_folder,
            columns=columns,
            rows=rows,
            intrinsics_path=intrinsics_path,
            undistort=undistort,
            display=display,
            save_images_with_overlayed_detected_corners=(
                save_images_with_overlayed_detected_corners
            ),
        )
    if board_type == BoardType.CHARUCO:
        return detect_charuco(
            images_parent_folder=images_parent_folder,
            charuco_detector=charuco_detector,
            columns=columns,
            rows=rows,
            intrinsics_path=intrinsics_path,
            undistort=undistort,
            display=display,
            save_images_with_overlayed_detected_corners=(
                save_images_with_overlayed_detected_corners
            ),
        )
    raise ValueError("board_type must be either 'chessboard' or 'charuco'")


def _camera_folders(images_parent_folder: str) -> Dict[str, str]:
    return {
        cam: os.path.join(images_parent_folder, cam)
        for cam in os.listdir(images_parent_folder)
        if os.path.isdir(os.path.join(images_parent_folder, cam))
    }


def _load_intrinsics(
    intrinsics_path: str,
    cam: str,
    use_dir: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if use_dir:
        per_cam_file = os.path.join(
            intrinsics_path, f"{cam}_intrinsics.json"
        )
        return load_intrinsics(per_cam_file, cam=None)
    return load_intrinsics(intrinsics_path, cam=cam)


def detect_chessboards(
    images_parent_folder: str,
    columns: int,
    rows: int,
    intrinsics_path: Optional[str] = None,
    undistort: bool = True,
    display: bool = False,
    save_images_with_overlayed_detected_corners: bool = True,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Detect chessboard corners per camera and frame.
    """
    image_folders = _camera_folders(images_parent_folder)
    use_dir = os.path.isdir(intrinsics_path) if intrinsics_path else False

    correspondences: Dict[str, Dict[int, np.ndarray]] = {}
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )
    corner_sub_pix_win_size = 5

    for cam, image_folder in tqdm(
        image_folders.items(),
        desc="Processing cameras",
        total=len(image_folders),
        leave=False,
    ):
        correspondences[cam] = {}
        if save_images_with_overlayed_detected_corners:
            out_dir = os.path.join(image_folder, "extracted_corners")
            os.makedirs(out_dir, exist_ok=True)

        if undistort and intrinsics_path:
            camera_matrix, distortion_coeffs = _load_intrinsics(
                intrinsics_path, cam, use_dir
            )
        else:
            camera_matrix, distortion_coeffs = None, None  # type: ignore

        files = [
            f for f in os.listdir(image_folder)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        files.sort(key=lambda x: int(x.split(".")[0]))

        for filename in tqdm(
            files, desc=f"Processing camera {cam}",
            total=len(files), leave=False
        ):
            k = int(filename.split(".")[0])
            file_path = os.path.join(image_folder, filename)

            img = cv.imread(file_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(
                gray,
                (columns, rows),
                cv.CALIB_CB_ADAPTIVE_THRESH
                | cv.CALIB_CB_FAST_CHECK
                | cv.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not ret:
                continue

            corners2 = np.squeeze(
                cv.cornerSubPix(
                    gray,
                    corners,
                    (corner_sub_pix_win_size, corner_sub_pix_win_size),
                    (-1, -1),
                    criteria,
                )
            )

            if display:
                cv.drawChessboardCorners(img, (columns, rows), corners2, ret)
                cv.imshow("img", img)
                cv.waitKey(1)
                if save_images_with_overlayed_detected_corners:
                    cv.imwrite(os.path.join(out_dir, filename), img)

            if undistort and camera_matrix is not None:
                points_reshaped = corners2.reshape(-1, 1, 2)
                undistorted_points = cv.undistortPoints(
                    points_reshaped, camera_matrix, distortion_coeffs,
                    P=camera_matrix,
                )
                _2dpoints = undistorted_points.reshape(-1, 2)
            else:
                _2dpoints = corners2

            correspondences[cam][k] = _2dpoints

    return correspondences


def detect_charuco(
    images_parent_folder: str,
    charuco_detector: cv.aruco.CharucoDetector,
    columns: int,
    rows: int,
    intrinsics_path: Optional[str] = None,
    undistort: bool = True,
    display: bool = False,
    save_images_with_overlayed_detected_corners: bool = True,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Detect ChArUco corners per camera and frame.
    """
    image_folders = _camera_folders(images_parent_folder)
    use_dir = os.path.isdir(intrinsics_path) if intrinsics_path else False

    correspondences: Dict[str, Dict[int, np.ndarray]] = {}

    for cam, image_folder in tqdm(
        image_folders.items(),
        desc="Processing cameras",
        total=len(image_folders),
        leave=False,
    ):
        correspondences[cam] = {}
        if save_images_with_overlayed_detected_corners:
            out_dir = os.path.join(image_folder, "extracted_corners")
            os.makedirs(out_dir, exist_ok=True)

        if undistort and intrinsics_path:
            camera_matrix, distortion_coeffs = _load_intrinsics(
                intrinsics_path, cam, use_dir
            )
        else:
            camera_matrix, distortion_coeffs = None, None  # type: ignore

        files = [
            f for f in os.listdir(image_folder)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        files.sort(key=lambda x: int(x.split(".")[0]))

        for filename in tqdm(
            files, desc=f"Processing camera {cam}",
            total=len(files), leave=False
        ):
            k = int(filename.split(".")[0])
            file_path = os.path.join(image_folder, filename)

            img = cv.imread(file_path)
            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,  # noqa: F841
            ) = charuco_detector.detectBoard(img)

            if display and charuco_ids is not None:
                cv.imshow("img", img)
                cv.waitKey(1)
                if save_images_with_overlayed_detected_corners:
                    cv.imwrite(os.path.join(out_dir, filename), img)

            if charuco_corners is None or charuco_ids is None:
                continue

            if undistort and camera_matrix is not None:
                points_reshaped = charuco_corners.reshape(-1, 1, 2)
                undistorted_points = cv.undistortPoints(
                    points_reshaped, camera_matrix, distortion_coeffs,
                    P=camera_matrix,
                )
                pts_2d = undistorted_points.reshape(-1, 2)
            else:
                pts_2d = charuco_corners.reshape(-1, 2)

            dense = np.full((columns * rows, 2), np.nan, dtype=float)
            ids = np.array(charuco_ids).reshape(-1)
            for i, cid in enumerate(ids):
                dense[cid, :] = pts_2d[i, :]

            correspondences[cam][k] = dense

    return correspondences
