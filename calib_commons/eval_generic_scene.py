from typing import Dict
import numpy as np
from scipy import stats
import json
import copy

from calib_commons.scene import Scene
from calib_commons.observation import Observation
from calib_commons.types import idtype
from calib_commons.utils.utils import view_score


def eval_reprojection_error(errors_x_y: np.ndarray):
    """
    Calculate the mean and standard deviation of reprojection errors.
    
    Parameters:
    reprojections_array (np.ndarray): Array of reprojections, shape (n, 2)
    observations_array (np.ndarray): Array of observed points, shape (n, 2)
    
    Returns:
    mean_error (float): Mean of reprojection errors
    std_error (float): Standard deviation of reprojection errors
    """

    if len(errors_x_y) == 0:
        return None, None
    errors = np.linalg.norm(errors_x_y, axis=1)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error, std_error


def get_coords_observations_paired_with_reprojections(scene: Scene, observations: Dict[int, Dict[int, Observation]], cam_id):
    obsv_coords = {}
    repr_coords = {}
    for point_id, observation in observations[cam_id].items():
        if observation._is_conform:
            if scene.reprojections[cam_id].get(point_id): 
                obsv_coords[point_id] = observation._2d 
                repr_coords[point_id] = scene.reprojections[cam_id][point_id]._2d
    return obsv_coords, repr_coords
        
def compute_reprojections_errors_x_y_in_cam(cam_id: idtype, scene: Scene, observations: Dict[int, Dict[int, Observation]], which=None):
    assert (which is not None), "arg named 'which' must be provided"
    
    # obsv_coords = []
    # repr_coords = []
    errors = []

    if which == "conform":
        for point_id, observation in observations[cam_id].items():
            if observation._is_conform:
                if scene.reprojections[cam_id].get(point_id): 
                    reprojection = scene.reprojections[cam_id][point_id]._2d
                    error = observation._2d - reprojection
                    errors.append(error)
                    # obsv_coords.append(observation._2d)
                    # repr_coords.append(reprojection)
    elif which == "non-conform":
        for point_id, observation in observations[cam_id].items():
            if not observation._is_conform:
                if scene.reprojections[cam_id].get(point_id): 
                    reprojection = scene.reprojections[cam_id][point_id]._2d
                    error = observation._2d - reprojection
                    errors.append(error)
                    # obsv_coords.append(observation._2d)
                    # repr_coords.append(reprojection)
    elif which == "all":
        for point_id, observation in observations[cam_id].items():
            if scene.reprojections[cam_id].get(point_id): 
                reprojection = scene.reprojections[cam_id][point_id]._2d
                error = observation._2d - reprojection
                errors.append(error)
                # obsv_coords.append(observation._2d)
                # repr_coords.append(reprojection)
    return errors#, obsv_coords, repr_coords
                

def compute_errors_x_y(generic_scene: Scene, generic_observations: Dict[int, Dict[int, Observation]], camera_groups=None):
    """
    Compute the reprojections error (x,y) for each camera and camera group, based on a generic scene and generic observations.
    
    Parameters:
    generic_scene (GenericScene): The scene containing cameras and reprojections
    generic_observations (Correspondences): Observations of the scene in the form of {camera_id: {point_id: Observation}}
    camera_groups (dict or None): Dictionary of camera groups {group_id: [camera_ids]}, optional
    
    Returns:
    result (dict): Dictionary with overall, per camera, and per camera group reprojection errors
    """

    all_errors = []
    # errors_dict = {}
    # Per-camera results
    per_camera = {}
    
    for cam_id in generic_observations.keys():
        errors = compute_reprojections_errors_x_y_in_cam(cam_id, generic_scene, generic_observations, which="conform")
        errors_array = np.array(errors)
        per_camera[cam_id] = errors_array
        if len(errors_array) > 0:
            all_errors.append(errors_array)
    if all_errors:
        all_errors = np.concatenate(all_errors, axis=0)
    else: 
        all_errors = None
    
    # Camera group results
    if camera_groups is not None:
        per_camera_group = {}
        for group_id, cam_ids in camera_groups.items():
            group_errors = []
            for cam_id in cam_ids:
                if cam_id in per_camera:
                    if len(per_camera[cam_id]) > 0:
                        group_errors.append(per_camera[cam_id])
           
            if group_errors:
                group_errors = np.concatenate(group_errors, axis=0)
                per_camera_group[group_id] = group_errors
                
    else:
        per_camera_group = None
    
    # Final result dictionary
    result = {
        'overall': all_errors,
        'per_camera': per_camera,
        'per_camera_group': per_camera_group
    }
    
    return result
    
   
def computes_metrics_for_groups(errors, view_scores, n_corres, camera_groups):


   
    errors_per_camera_group = {}
    for group_id, cam_ids in camera_groups.items():
        group_errors = []
        for cam_id in cam_ids:
            if cam_id in errors:
                if len(errors[cam_id]) > 0:
                    group_errors.append(errors[cam_id])
        
        if group_errors:
            group_errors = np.concatenate(group_errors, axis=0)
            errors_per_camera_group[group_id] = group_errors


    res_per_camera_group = {}
    for group_id, cam_ids in camera_groups.items():
        if group_id in errors_per_camera_group:
            group_error_mean, group_error_std = eval_reprojection_error(errors_per_camera_group[group_id])
            group_mean_view_score = np.mean([view_scores[cam_id] for cam_id in cam_ids if cam_id in view_scores])
            group_mean_n_corres = np.mean([n_corres[cam_id] for cam_id in cam_ids if cam_id in n_corres])
        else:
            group_error_mean, group_error_std = None, None
            group_mean_view_score = None
            group_mean_n_corres = None
        res_per_camera_group[group_id] = {'mean_error': group_error_mean, 
                                          'std_error': group_error_std, 
                                          'mean_view_score': group_mean_view_score, 
                                          'mean_n_corres': group_mean_n_corres}
    
    res_per_camera = {}
    for cam_id in camera_groups['all']: 
        if cam_id in errors:
            cam_error_mean, cam_error_std = eval_reprojection_error(errors[cam_id])
            res_per_camera[cam_id] = {'mean_error': cam_error_mean, 'std_error': cam_error_std, 'view_score': view_scores[cam_id], 'n_corres': n_corres[cam_id]}
        else:
            res_per_camera[cam_id] = {'mean_error': None, 'std_error': None, 'view_score': None, 'n_corres': None}


    return res_per_camera_group, res_per_camera

def eval_generic_scene(generic_scene: Scene, generic_observations: Dict[int, Dict[int, Observation]], camera_groups=None, save_to_json=True, output_path=None, success_pixel_threshold=None, print_=False):
    """
    Evaluate the reprojection error for each camera and camera group, based on a generic scene and generic observations.
    
    Parameters:
    generic_scene (GenericScene): The scene containing cameras and reprojections
    generic_observations (Correspondences): Observations of the scene in the form of {camera_id: {point_id: Observation}}
    camera_groups (dict or None): Dictionary of camera groups {group_id: [camera_ids]}, optional
    
     result (dict): Dictionary with overall, per camera, and per camera group (mean, std) reprojection errors
    
    """
    
    if save_to_json: 
        assert output_path is not None, "output_path must be provided if save_to_json is True"

    camera_groups_raw = copy.deepcopy(camera_groups)

    # reprojection error
    errors_dict = compute_errors_x_y(generic_scene, generic_observations, camera_groups)
    errors_x_y_per_camera = errors_dict['per_camera']

    if 0:
        
        import matplotlib.pyplot as plt

        # Determine the number of rows and columns for the grid
        num_cameras = len(errors_x_y_per_camera)
        num_cols = 3
        num_rows = (num_cameras + num_cols - 1) // num_cols

        # Create a figure with subplots for each camera in a grid
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for ax, (cam_id, errors) in zip(axes, errors_x_y_per_camera.items()):
            if len(errors) > 0:
                euclidean_errors = np.linalg.norm(errors, axis=1)
                ax.hist(euclidean_errors, bins=50, edgecolor='black')
                ax.set_title(f'Camera {cam_id} Euclidean Error Histogram')
                ax.set_xlabel('Euclidean Error')
                ax.set_ylabel('Frequency')
                ax.grid(True)

        # Hide any unused subplots
        for i in range(len(errors_x_y_per_camera), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()
        

    view_scores = {}
    n_corres = {}
    for cam_id in generic_scene.cameras: 
        observations_coords_dict, _ = get_coords_observations_paired_with_reprojections(generic_scene, generic_observations, cam_id)
        observations_coords = np.array(list(observations_coords_dict.values()))
        view_scores[cam_id] = view_score(observations_coords, generic_scene.cameras[cam_id].intrinsics.resolution)
        n_corres[cam_id] = len(observations_coords)
    
    if camera_groups is None:
        camera_groups = {}

    camera_groups_ = copy.deepcopy(camera_groups)   
    camera_groups_['all'] = list(generic_scene.cameras.keys())
    res_per_camera_group, res_per_camera = computes_metrics_for_groups(errors_x_y_per_camera, view_scores, n_corres, camera_groups_)

    # print(res_per_camera_group)
    # print(res_per_camera)

    all_cameras_to_reconstruct = []
    for group_id, cam_ids in camera_groups.items():
        all_cameras_to_reconstruct.extend(cam_ids)

    if success_pixel_threshold is not None:
        
        n_cameras_to_reconstruct = len(all_cameras_to_reconstruct)
        n_cameras_reconstructed = len(res_per_camera)
        cameras_successfully_reconstructed = []
        for cam_id, metrics in res_per_camera.items():
            if metrics['mean_error'] is not None and metrics['mean_error'] < success_pixel_threshold:
                cameras_successfully_reconstructed.append(cam_id)
        n_cameras_successfully_reconstructed = len(cameras_successfully_reconstructed)
        # # n_cameras_successfully_reconstructed = sum(1 for cam_id, metrics in res_per_camera.items() if metrics['mean_error'] is not None and metrics['mean_error'] < 0.5)
        # print(f"Number of cameras successfully reconstructed with mean error < 0.5: {n_cameras_successfully_reconstructed}")

        # n_cameras_successfully_reconstructed = 
        success_rate = n_cameras_successfully_reconstructed / n_cameras_to_reconstruct * 100
        success_rate_res = {'n_cameras_to_reconstruct': n_cameras_to_reconstruct, 'n_cameras_reconstructed': n_cameras_successfully_reconstructed, 'success_rate': success_rate}
    else: 
        success_rate_res = {'n_cameras_to_reconstruct': None, 'n_cameras_reconstructed': None, 'success_rate': None}

        
    # Prepare data for JSON
    result = {
        'per_camera_group': res_per_camera_group,
        'per_camera': res_per_camera, 
        'success_rate': success_rate_res
    }

    if save_to_json:

        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Results saved to {output_path}")


    # import matplotlib.pyplot as plt

    # Plot histogram of the norm2 error of errors_dict['overall']
    # mean_error, std_error = eval_reprojection_error(errors_dict['overall'])
    # Calculate the mode of the norm2 errors
    # norm2_errors = np.linalg.norm(errors_dict['overall'], axis=1)
    # median_error = np.median(norm2_errors)

    # Plot the mode
    # plt.axvline(median_error, color='b', linestyle='dashed', linewidth=1, label=f'Median: {median_error:.2f}')
    # plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_error:.2f}')
    # plt.axvline(mean_error + std_error, color='g', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_error:.2f}')
    # plt.axvline(mean_error - std_error, color='g', linestyle='dashed', linewidth=1)
    # plt.legend()
    # plt.hist(np.linalg.norm(errors_dict['overall'], axis=1), bins=100, edgecolor='black')
    # plt.title('Histogram of Norm2 Errors')
    # plt.xlabel('Norm2 Error')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()



    if print_:
        print("")
        print_nicely_eval_results(result, print_success_rate=success_pixel_threshold is not None)

    return result

def print_nicely_eval_results(eval_results: dict, print_success_rate=True) -> None:
    """
    Prints the evaluation metrics in a nicely formatted way.

    Args:
        eval_results (dict): The evaluation results from eval_generic_scene.
    """
    def print_dict(d: dict, indent: int = 0):
        for key, value in d.items():
            if key == 'success_rate' and not print_success_rate:
                continue
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                print_dict(value, indent + 4)
            else:
                if isinstance(value, float):
                    value = f"{value:.3f}"
                print(" " * indent + f"{key}: {value}")

    print("Evaluation Metrics:")
    print("===================")
    print_dict(eval_results)
    
    # # overall 
    # res_overall = eval_reprojection_error(errors_dict['overall'])

    # # per camera
    # res_per_camera = {cam_id: eval_reprojection_error(errors) for cam_id, errors in errors_dict['per_camera'].items()} 

    # # per camera group
    # res_per_camera_group = {group_id: eval_reprojection_error(errors) for group_id, errors in errors_dict['per_camera_group'].items()}









# def save_eval_results_to_json(results, filename, n_digits=2):
#     """
#     Save the results of eval_generic_scene into a JSON file, with values rounded to the specified number of decimal places.

#     Parameters:
#     results (dict): A dictionary containing the results from eval_generic_scene
#     filename (str): The output file name for saving the results.
#     n_digits (int): Number of decimal places to round to (default is 2).
#     """
    
#     # Convert the results dictionary into a JSON-friendly format, rounding to n_digits decimal places
#     json_friendly_results = {
#         'overall': {
#             'mean_error': round(results['overall'][0], n_digits),
#             'std_error': round(results['overall'][1], n_digits)
#         },
#         'per_camera': {
#             cam_id: {
#                 'mean_error': round(res[0], n_digits),
#                 'std_error': round(res[1], n_digits)
#             } for cam_id, res in results['per_camera'].items()
#         },
#         'per_camera_group': {
#             group_id: {
#                 'mean_error': round(res[0], n_digits),
#                 'std_error': round(res[1], n_digits)
#             } for group_id, res in results['per_camera_group'].items()
#         }
#     }
    
#     # Save the result to a JSON file
#     with open(filename, 'w') as f:
#         json.dump(json_friendly_results, f, indent=4)
#     print(f"Results saved to {filename}")