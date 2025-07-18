import os
import random
import cv2
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import subprocess
from typing import Optional

from calib_commons.utils.utils import blur_score

def save_calibration_to_json(mtx, dist, filename):
    """Saves camera calibration data (matrix and distortion) to a JSON file."""
    calibration_data = {
        "sensors": {
            "RGB": {
                "intrinsics": {
                    "type_id": "opencv-matrix",
                    "rows": mtx.shape[0],
                    "cols": mtx.shape[1],
                    "dt": "d",
                    "data": mtx.flatten().tolist()
                },
                "distortionCoefficients": {
                    "type_id": "opencv-matrix",
                    "rows": dist.shape[0],
                    "cols": dist.shape[1],
                    "dt": "d",
                    "data": dist.flatten().tolist()
                }
            }
        }
    }

    with open(filename, 'w') as file:
        json.dump(calibration_data, file, indent=4)

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    sampling_step: int,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> None:
    """
    Extracts frames from a video file using the ffmpeg command-line tool.

    This function attempts to use CUDA hardware acceleration if available to
    speed up the process.

    Args:
        video_path: The path to the input video file.
        output_dir: The directory where the extracted frames will be saved.
        sampling_step: The frame sampling rate (e.g., 5 means every 5th frame).
        start_time: The optional start time for extraction (format: "hh:mm:ss.ms").
        end_time: The optional end time for extraction (format: "hh:mm:ss.ms").
    """
    # Build time-trimming argument string
    trim_txt = ""
    if start_time:
        trim_txt += f"-ss {start_time} "
    if end_time:
        trim_txt += f"-to {end_time} "

    # Check if CUDA hardware acceleration is available
    hwaccel_available = False
    try:
        cmd = "ffmpeg -hide_banner -hwaccels | grep cuda"
        encoders = subprocess.check_output(cmd, shell=True, text=True)
        if "cuda" in encoders:
            hwaccel_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "WARNING: cuda hwaccel not available, frame extraction will be slow. "
            "Install NVIDIA drivers and ffmpeg with cuda support for better speeds."
        )

    hwaccel = " -hwaccel cuda" if hwaccel_available else ""

    # Construct the final command string
    # Note: The output format %04d.jpg will create sequentially numbered frames
    # (0001.jpg, 0002.jpg, ...), which is suitable for the calibration process.
    ffmpeg_cmd = (
        f"ffmpeg{hwaccel} {trim_txt}-i \"{video_path}\" "
        f'-vf "select=not(mod(n\\,{sampling_step}))" -vsync vfr '
        f"\"{os.path.join(output_dir, '%04d.jpg')}\""
    )

    # Ensure the output directory exists and run the command
    os.makedirs(output_dir, exist_ok=True)
    print(f"Executing FFmpeg command: {ffmpeg_cmd}")
    subprocess.run(ffmpeg_cmd, shell=True, check=True)

def calibrate_camera_from_images(
        images_path,
        square_size,
        col,
        row,
        debug=True,
        error_threshold=3,
        max_frames=150
    ):
    """Calibrates the camera using chessboard images."""
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corner_sub_pix_win_size = 11

    objpoints, imgpoints = [], []
    files = [os.path.join(images_path, f) for f in os.listdir(images_path) if (f.lower().endswith(('png', 'jpg', 'jpeg')))]

    if not files:
        print(f"Warning: No images found in {images_path}. Skipping.")
        return None, None, None

    n_frames = min(len(files), max_frames)

    # Consider least blurry images
    files = sorted(files, key=blur_score, reverse=True)[:n_frames]

    for file_path in tqdm(files, desc="Processing images"):
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read image {file_path}. Skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (col, row))
        if ret:
            corners2 = np.squeeze(cv2.cornerSubPix(gray, corners, (corner_sub_pix_win_size, corner_sub_pix_win_size), (-1, -1), criteria))
            objpoints.append(objp)
            imgpoints.append(corners2)

            if debug:
                img = cv2.drawChessboardCorners(img, (col, row), corners2, ret)

        if debug:
            h, w = img.shape[:2]
            scale = 0.3
            disp_w, disp_h = int(w * scale), int(h * scale)
            cv2.namedWindow('Image with Corners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image with Corners', disp_w, disp_h)
            cv2.imshow('Image with Corners', img)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise ValueError("No valid chessboard patterns were detected.")

    print(f"Detected chessboard patterns in {len(objpoints)} out of {len(files)} images.")

    ret, mtx, dist, _, _, _, _, perViewError = cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

    if debug:
        plt.hist(perViewError)
        plt.title("Reprojection Errors per View")
        plt.xlabel("Error (pixels)")
        plt.ylabel("Frequency")
        plt.show()

    while np.any(perViewError > error_threshold):
        keep = np.where(perViewError <= error_threshold)[0]
        print(f"Found {len(objpoints) - len(keep)} outlier(s), rerunning calibration")
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        
        if not objpoints:
            raise ValueError("All points were considered outliers. Calibration failed.")

        ret, mtx, dist, _, _, _, _, perViewError = cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Camera calibrated with reprojection error (RMSE): {ret:.2f} [pix]")

    return ret, mtx, dist

def main():
    """Main function to run the camera calibration process."""
    parser = argparse.ArgumentParser(description="Camera calibration script using videos or images.")
    parser.add_argument("--use_videos", action="store_true", help="Set this flag to use videos instead of images.")
    parser.add_argument("--data_directory", type=str, default=None, help="Path to the data directory containing videos or image folders.")
    parser.add_argument("--output_parent_directory", type=str, default=None, help="Path to the directory where the output folder will be created.")
    parser.add_argument("--square_size", type=float, required=True, help="Size of a chessboard square in meters.")
    parser.add_argument("--chessboard_width", type=int, required=True, help="Number of inner corners in chessboard width.")
    parser.add_argument("--chessboard_height", type=int, required=True, help="Number of inner corners in chessboard height.")
    parser.add_argument("--sampling_step", type=int, default=45, help="Frame sampling rate (e.g., every 45th frame). Used only with --use_videos.")
    parser.add_argument("--debug", action="store_true", help="Show detected corners on the images during calibration.")

    args = parser.parse_args()

    data_directory = args.data_directory or os.getcwd()
    output_parent_directory = args.output_parent_directory or data_directory

    output_subfolder_name = "calibrate_intrinsics_output"
    output_directory = os.path.join(output_parent_directory, output_subfolder_name)
    os.makedirs(output_directory, exist_ok=True)

    intrinsics_dir = os.path.join(output_directory, "camera_intrinsics")
    os.makedirs(intrinsics_dir, exist_ok=True)

    images_directory = data_directory

    if args.use_videos:
        sampled_frames_dir = os.path.join(output_directory, "sampled_frames")
        os.makedirs(sampled_frames_dir, exist_ok=True)
        images_directory = sampled_frames_dir # The rest of the script will use frames from here

        video_files = [f for f in os.listdir(data_directory) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]

        if not video_files:
            print(f"Error: --use_videos flag is set, but no video files were found in {data_directory}")
            return

        for video_file in video_files:
            camera_name = Path(video_file).stem
            camera_output_dir = os.path.join(sampled_frames_dir, camera_name)
            os.makedirs(camera_output_dir, exist_ok=True)

            video_path = os.path.join(data_directory, video_file)
            print(f"\nExtracting frames from {video_file} into {camera_output_dir}...")
            # This now calls the FFmpeg-based function
            extract_frames_from_video(video_path, camera_output_dir, args.sampling_step)
        print("\nFrame extraction complete.")

    # Calibration process uses the `images_directory` which points to either the
    # original data directory or the newly created `sampled_frames` directory.
    folders_to_calibrate = [d for d in os.listdir(images_directory) if os.path.isdir(os.path.join(images_directory, d))]
    
    if not folders_to_calibrate:
        print(f"Error: No subdirectories to process in '{images_directory}'.")
        print("Please ensure your images/videos are organized in subdirectories, one for each camera.")
        return

    for folder in folders_to_calibrate:
        if folder == output_subfolder_name:
            continue
        folder_path = os.path.join(images_directory, folder)
        
        print(f"\n--- Calibrating camera for '{folder}' ---")
        try:
            ret, mtx, dist = calibrate_camera_from_images(
                folder_path,
                args.square_size,
                args.chessboard_width,
                args.chessboard_height,
                args.debug
            )
            if ret is not None:
                intrinsics_file = os.path.join(intrinsics_dir, f'{folder}_intrinsics.json')
                save_calibration_to_json(mtx, dist, intrinsics_file)
                print(f"Calibration data saved for '{folder}' to {intrinsics_file}")
        except (ValueError, RuntimeError) as e:
            print(f"Could not calibrate camera for '{folder}'. Reason: {e}")

if __name__ == "__main__":
    main()