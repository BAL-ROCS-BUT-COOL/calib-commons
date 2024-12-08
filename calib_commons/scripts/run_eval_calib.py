import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from calib_commons.utils.detect_board import detect_board_corners, BoardType
from calib_commons.utils.calib_eval import CalibEvaluator
from calib_commons.utils.calib_eval_config import CalibEvaluatorConfig
from calib_commons.data.data_pickle import save_to_pickle, load_from_pickle

from calib_commons.correspondences import convert_correspondences_nparray_to_correspondences
from calib_commons.data.load_calib import construct_cameras_intrinsics
from calib_commons.data.data_json import load_cameras_poses_from_json
from calib_commons.eval_generic_scene import eval_generic_scene


from calib_commons.viz import visualization

############################### USER INTERFACE ####################################

# PATHS
images_parent_folder = str(Path(r"C:\Users\timfl\Documents\test_boardCal\eval_data"))
intrinsics_folder = str(Path(r"C:\Users\timfl\Documents\Master Thesis\Final_XP\intrinsics_calibration\output\gopro_240fps"))
camera_poses_estimate_json_path = Path(r"C:\Users\timfl\OneDrive - ETH Zurich\My_Documents\01_Etudes\Master\Master Thesis\Source\Release\calib-board\results\camera_poses.json")

# PRE-PROCESSING PARAMETERS
checker_rows = 4 # internal rows
checker_columns = 6 # internal columns
square_size = 0.165  # [m]

board_type = BoardType.CHARUCO
if board_type == BoardType.CHARUCO:
    charuco_marker_size = 0.123
    charuco_dictionary = cv2.aruco.DICT_4X4_100
show_detection_images = False
save_detection_images = False


# CALIBRATION PARAMETERS
eval_calib_config = CalibEvaluatorConfig(
    reprojection_error_threshold=1,
    ba_least_square_ftol=1e-8,
    display=False
)
out_folder_eval = Path("results")
show_viz = 1
save_viz = 0
save_eval_metrics_to_json = 1


###################### PRE-PROCESSING: CORNERS DETECTION ###########################

if board_type == BoardType.CHARUCO:
    charuco_detector = cv2.aruco.CharucoDetector(cv2.aruco.CharucoBoard((checker_columns+1, checker_rows+1), 
                                                                        square_size, 
                                                                        charuco_marker_size, 
                                                                        cv2.aruco.getPredefinedDictionary(charuco_dictionary)))
else: 
    charuco_detector = None

correspondences_nparray = detect_board_corners(images_parent_folder=images_parent_folder,
                                        board_type=board_type,
                                        charuco_detector=charuco_detector,
                                        columns=checker_columns, 
                                        rows=checker_rows, 
                                        intrinsics_folder=intrinsics_folder, 
                                        undistort=True, 
                                        display=show_detection_images, 
                                        save_images_with_overlayed_detected_corners=save_detection_images)
eval_correspondences = convert_correspondences_nparray_to_correspondences(correspondences_nparray)

# save_to_pickle(out_folder_eval / "eval_correspondences.pkl", eval_correspondences)
# eval_correspondences = load_from_pickle("results/eval_correspondences.pkl")


###################### EVAL CALIB ###########################

out_folder_eval.mkdir(parents=True, exist_ok=True)
intrinsics = construct_cameras_intrinsics(images_parent_folder, intrinsics_folder)
camera_poses_estimate = load_cameras_poses_from_json(camera_poses_estimate_json_path)


calib_evaluator = CalibEvaluator(eval_correspondences, intrinsics, camera_poses_estimate, eval_calib_config)
calib_evaluator.compute_initial_estimates()
calib_evaluator.iterative_filtering()
generic_scene = calib_evaluator.get_scene()
generic_obsv = calib_evaluator.correspondences

# Save files

scene_estimate_file = out_folder_eval / "scene_estimate.pkl"
save_to_pickle(scene_estimate_file, generic_scene)
print("scene estimate saved to", scene_estimate_file)

correspondences_file = out_folder_eval / "correspondences.pkl"
save_to_pickle(correspondences_file, generic_obsv)
print("correspondences saved to", correspondences_file)

metrics = eval_generic_scene(generic_scene, generic_obsv, camera_groups=None, save_to_json=save_eval_metrics_to_json, output_path=out_folder_eval / "metrics.json", print_ = True)
print("")

# Visualization
if show_viz or save_viz:
    dpi = 300
    save_path = out_folder_eval / "scene.png"
    visualization.visualize_scenes([generic_scene], show_ids=False, show_fig=show_viz, save_fig=save_viz, save_path=save_path)
    if save_viz:
        print("scene visualization saved to", save_path)
    save_path = out_folder_eval / "2d.png"
    visualization.visualize_2d(generic_scene, generic_obsv, which="both", show_ids=False, show_fig=show_viz, save_fig=save_viz, save_path=save_path)
    if save_viz:
        print("2d visualization saved to", save_path)
    save_path = out_folder_eval / "2d_errors.png"
    visualization.plot_reprojection_errors(scene_estimate=generic_scene, 
                                        observations=generic_obsv, 
                                        show_fig=show_viz,
                                        save_fig=save_viz, 
                                        save_path=save_path)
    if save_viz:
        print("2d errors visualization saved to", save_path)

    if show_viz:
        plt.show()
