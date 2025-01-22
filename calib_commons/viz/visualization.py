from typing import List

import matplotlib.pyplot as plt
from matplotlib import rcParams
import math 
import numpy as np

from calib_commons.types import idtype
from calib_commons.correspondences import Correspondences
from calib_commons.scene import Scene, SceneType
from calib_commons.eval_generic_scene import compute_reprojections_errors_x_y_in_cam

from calib_commons.viz.visualization_tools import get_color_from_error, get_color_from_id
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os



MODIFIERS = {SceneType.SYNTHETIC: r"\overline", 
             SceneType.ESTIMATE: r"\hat"}

MARKERS_OBSERVATIONS = {SceneType.SYNTHETIC: '+', 
                       SceneType.ESTIMATE: 'x'}
MARKERS_3D_POINTS = {SceneType.SYNTHETIC: 'o', 
                       SceneType.ESTIMATE: '.'}


# XLIM = [-1,1]
# YLIM = XLIM
# ZLIM = XLIM
def get_coords(scene: Scene, x,y,z, points_ids): 

    for camera in scene.cameras.values(): 
        x.append(camera.pose.get_x())
        y.append(camera.pose.get_y())
        z.append(camera.pose.get_z())

    for point in scene.object_points.values(): 
        if points_ids is None or point.id in points_ids:
            x.append(point.position[0])
            y.append(point.position[1])
            z.append(point.position[2])
    
def init_ax(ax): 
    ax.set_aspect("auto")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def visualize_scenes(scenes: List[Scene], show_ids = True, show_only_points_with_estimate = False,
                show_fig = True, 
                save_fig = False, 
                save_path = None, 
                dpi = 300, 
                elev=None, 
                azim = None,
                colors = None, 
                frustum_scaling = 1) -> None: 
    fig = plt.figure(figsize=(6, 6)) 
    rcParams['text.usetex'] = True
    ax = fig.add_subplot(projection='3d')

    
    points_ids = None 
    if show_only_points_with_estimate:
        for scene in scenes:
            if scene.type == SceneType.ESTIMATE: 
                points_ids = list(scene.object_points.keys())


    # xmin = np.inf
    # ymin = np.inf
    # zmin = np.inf

    # xmax = -np.inf
    x = []
    y = []
    z = []

    for scene in scenes: 
        get_coords(scene, x,y,z, points_ids)

    x_min = min(x)
    x_max = max(x)

    y_min = min(y)
    y_max = max(y)

    z_min = min(z)
    z_max = max(z)

    
        
    min_coords = min([x_min, y_min, z_min])
    max_coords = max([x_max, y_max, z_max])

    # print("x: ", x_min, x_max)
    # print("y: ", y_min, y_max)
    # print("z: ", z_min, z_max)
    size_x = x_max-x_min
    size_y = y_max-y_min
    size_z = z_max-z_min

    size = max([size_x, size_y, size_z])
    # print("size ", size)
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2

    xlim = [x_mid-size/2, x_mid+size/2]
    ylim = [y_mid-size/2, y_mid+size/2]
    zlim = [z_mid-size/2, z_mid+size/2]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')   
    ax.set_box_aspect([1, 1, 1]) 
    ax.set_title('3D Scenes', fontsize=14, pad=50)   

    if elev and azim:
        ax.view_init(elev=elev, azim=azim)  # Change elev and azim as needed
    # ax.grid(False)
    # ax.set_axis_off()  # This removes the axis lines, ticks, and labels
    # # Remove pane background color (make it transparent)
    # ax.xaxis.pane.fill = False

    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    # Remove pane background color (make it transparent)
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    for scene in scenes: 
        plot_scene(scene, xlim, ylim, zlim, show_ids, points_ids, frustum_scaling=frustum_scaling)

        if save_fig:
                # Create the directory if it does not exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show_fig:
        plt.show(block=False)

    # plt.show(block=False)

def visualize_2d(scene: Scene = None, 
                observations = None, 
                show_only_points_with_both_obsv_and_repr=True,
                show_ids = True, 
                which = "both",
                show_fig = True, 
                save_fig = False, 
                save_path = None, 
                dpi = 300) -> None: 
    
    #hey
    if not scene: 
        camerasId = observations.keys()

        
    camerasId = scene.cameras.keys() 

    # points_ids = None 
    # if show_only_points_with_both_obsv_and_repr: 
    #     # obsv_ids




    n = len(camerasId)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n/ cols)
    fig_width = cols * 4  # Each subplot has a width of 4 inches (adjust as needed)
    fig_height = rows * 3  # Each subplot has a height of 3 inches (adjust as needed)

    # Create a figure with dynamic size
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # fig, axs = plt.subplots(rows, cols)
    axs = np.array(axs).reshape(-1) if n > 1 else [axs]
    axsUsed = [False for ax in axs]

    axIndex = 0
    for cameraId in camerasId:

        points_ids = None
        if show_only_points_with_both_obsv_and_repr: 
            obsv_ids = set(observations[cameraId].keys())
            repr_ids = set(scene.reprojections[cameraId].keys())
            points_ids = obsv_ids & repr_ids
        else:
            points_ids = list(observations[cameraId].keys())

        _2d_observations_conform = {}
        _2d_observations_nonconform = {}

        _2d_reprojections_conform = {}
        _2d_reprojections_nonconform = {}

        for point_id in points_ids:
            if observations[cameraId][point_id]._is_conform:
                _2d_observations_conform[point_id] = observations[cameraId][point_id]._2d
                if scene.reprojections[cameraId].get(point_id):
                    _2d_reprojections_conform[point_id] = scene.reprojections[cameraId][point_id]._2d
            else:
                _2d_observations_nonconform[point_id] = observations[cameraId][point_id]._2d
                if scene.reprojections[cameraId].get(point_id):
                    _2d_reprojections_nonconform[point_id] = scene.reprojections[cameraId][point_id]._2d

        ax = axs[axIndex]
        axsUsed[axIndex] = True
        axIndex += 1
        
        resX = scene.cameras[cameraId].intrinsics.resolution[0]
        resY = scene.cameras[cameraId].intrinsics.resolution[1]
        ax.set_xlim((0, resX))
        ax.set_ylim((resY, 0))
        aspect_ratio = resY / resX
        ax.set_aspect(aspect_ratio / ax.get_data_ratio())


        title = 'Obsv/repro. in ' + str(cameraId)
        ax.set_title(title)

        
        if which == "both" or which == "conform":
            for point_id, _2d in _2d_observations_conform.items(): 
                _2d = _2d.squeeze()
                color = 'blue'
                alpha = 1
                marker = MARKERS_OBSERVATIONS[SceneType.SYNTHETIC]
                ax.plot(_2d[0], _2d[1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
                    
                if show_ids:
                    ax.text(_2d[0], _2d[1], point_id, color='black', fontsize=10, alpha=1)
                
            for point_id, _2d in _2d_reprojections_conform.items(): 
                color = 'blue'
                alpha = 1
                marker = MARKERS_OBSERVATIONS[SceneType.ESTIMATE]
                ax.plot(_2d[0], _2d[1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha) 

        if which == "both" or which == "non-conform":
            for point_id, _2d in _2d_observations_nonconform.items(): 
                color = 'red'
                alpha = 1
                _2d = _2d.squeeze()

                marker = MARKERS_OBSERVATIONS[SceneType.SYNTHETIC]
                ax.plot(_2d[0], _2d[1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)
                if show_ids:
                    ax.text(_2d[0,0], _2d[0,1], point_id, color='black', fontsize=10, alpha=1)  
                
            for point_id, _2d in _2d_reprojections_nonconform.items(): 
                color = 'red'
                alpha = 1
                _2d = _2d.squeeze()

                marker = MARKERS_OBSERVATIONS[SceneType.ESTIMATE]
                ax.plot(_2d[0], _2d[1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  

        
        
    for i, ax in enumerate(axs): 
        if not axsUsed[i]: 
            ax.axis('off')  # Turn off unused subplots

    if save_fig:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show_fig:
        plt.show(block=False)
        


        
def plot_reprojections_in_camera(correspondences, 
                             cameraId: idtype,
                             ax, 
                             cmap = None, 
                             reprojectionErrors = None, 
                             show_ids=True, 
                             points_ids=None) -> None:
    
    if not points_ids: 
        points_ids=correspondences[cameraId]

    if reprojectionErrors:
        errorsList = []
        for errorsOfCamera in reprojectionErrors.values():
            errorsList.extend(errorsOfCamera.values())

        if errorsList:
            errorMin = min(errorsList)
            errorMax = max(errorsList)
    marker = MARKERS_OBSERVATIONS[SceneType.ESTIMATE]

    for point_id in points_ids:
        observation = correspondences[cameraId][point_id]
   
        if cmap=="id":
            color = get_color_from_id(point_id)
        elif cmap=="reprojectionError": 
            error = reprojectionErrors[cameraId].get(point_id)
            if error:
                color = get_color_from_error(error, errorMin, errorMax)
            else:
                raise ValueError("no error found for this point.")
        else:
            raise ValueError("cmap not implemented.")
        alpha = 1
        # print(point_id)
        ax.plot(observation._2d[:,0], observation._2d[:,1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
        if show_ids:
            ax.text(observation._2d[0,0], observation._2d[0,1], point_id, color='black', fontsize=10, alpha=1)

    if cmap=="reprojectionError": 
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin=errorMin, vmax=errorMax))
        sm.set_array([])  # You can safely ignore this line.

        # Add the colorbar to the figure
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Reprojection Error')


        
    

def plot_observations_in_camera(correspondences, 
                             cameraId: idtype,
                             ax, 
                             cmap = None, 
                             reprojectionErrors = None, 
                             show_ids=True, 
                            points_ids=None) -> None:
    
    if not points_ids: 
        points_ids=correspondences[cameraId]

    for point_id in points_ids:
        observation = correspondences[cameraId][point_id]
        if observation and observation._is_conform:
        
            marker = MARKERS_OBSERVATIONS[SceneType.SYNTHETIC]
            alpha=0.5
            color = "black"
    
            ax.plot(observation._2d[:,0], observation._2d[:,1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
            if show_ids:
                ax.text(observation._2d[0,0], observation._2d[0,1], point_id, color=color, fontsize=10, alpha=1)

  

def plot_scene(scene: Scene, xlim, ylim, zlim, show_ids = True, points_ids = None, colors = None, frustum_scaling=1) -> None: 
    mod = MODIFIERS[scene.type]
    marker = MARKERS_3D_POINTS[scene.type]

    if points_ids == None:
        points_ids = list(scene.object_points.keys())
        

    ax = plt.gca() 
    for camera in scene.cameras.values(): 
        name = r"$\{" + mod + r"{C}_{" + str(camera.id) + r"}\}$"
        camera.plot(name, frustum_scaling=frustum_scaling)

    for point_id in points_ids:
        point = scene.object_points[point_id]
        color = None
        if point.color:
            color = point.color
        else:
            color=get_color_from_id(point.id)
        # ax.scatter(point.position[0], point.position[1], point.position[2,], color=color, s=2, marker='^')
        ax.scatter(point.position[0], point.position[1], point.position[2,], marker=marker, c=color, s=2)


        # ax.scatter(xs, ys, zs, marker=m)

        x = point.position[0]
        y = point.position[1]
        z = point.position[2]
        if show_ids:
            ax.text(x, y, z, str(point.id), color=get_color_from_id(point.id), fontsize=10)



def plot_reprojection_errors(scene_estimate: Scene, 
                             observations: Correspondences, 
                             show_fig = True,
                            save_fig = False, 
                            save_path = None, 
                            dpi = 300) -> None: 
    
    n = len(scene_estimate.cameras)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n/ cols)

    fig_width = cols * 4  # Each subplot has a width of 4 inches (adjust as needed)
    fig_height = rows * 4  # Each subplot has a height of 3 inches (adjust as needed)

    # Create a figure with dynamic size
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # fig, axs = plt.subplots(rows, cols)
    axs = np.array(axs).reshape(-1) if n > 1 else [axs]
    axsUsed = [False for ax in axs]

    axIndex = 0
    
    all_errors = []
    all_errors_conform = []

    errors_conform_per_camera = {}
    errors_nonconform_per_camera = {}

    for camera in scene_estimate.cameras.values(): 
        ax = axs[axIndex]
        axsUsed[axIndex] = True
        axIndex += 1
        
        errors_conform = compute_reprojections_errors_x_y_in_cam(camera.id, scene_estimate, observations, which="conform")
        errors_nonconform = compute_reprojections_errors_x_y_in_cam(camera.id, scene_estimate, observations, which="non-conform")

        errors_conform_per_camera[camera.id] = np.array(errors_conform)
        errors_nonconform_per_camera[camera.id] = np.array(errors_nonconform)

        errors_tot = np.concatenate(errors_conform+errors_nonconform, axis=0)        
        all_errors.extend(errors_tot)
        all_errors_conform.extend(errors_conform)
       
        aspect_ratio = 1
        ax.set_aspect(aspect_ratio / ax.get_data_ratio())
        title = f"Repr. errors in {camera.id}"
        ax.set_title(title)


    error_min = np.nanmin(all_errors_conform)
    error_max = np.nanmax(all_errors_conform)
    # plot_lim = max(abs(error_min), abs(error_max))
    plot_lim = 2
    axIndex = 0
    for camera in scene_estimate.cameras.values(): 
        ax = axs[axIndex]
        axIndex += 1
        ax.set_xlim([-plot_lim, plot_lim])
        ax.set_ylim([-plot_lim, plot_lim])
    
    # errors_conform_array = np.array(errors_conform)
    # errors_nonconform_array = np.array(errors_nonconform)

    axIndex = 0
    for camera in scene_estimate.cameras.values(): 
        ax = axs[axIndex]
        axIndex += 1

        # errors_ = errors_conform_per_camera[camera.id]
        if len(errors_conform_per_camera[camera.id]) > 0:
            ax.scatter(errors_conform_per_camera[camera.id][:,0], errors_conform_per_camera[camera.id][:,1], s=0.5, c='blue', marker = 'o', alpha=1)  
        if len(errors_nonconform_per_camera[camera.id]) > 0:
            ax.scatter(errors_nonconform_per_camera[camera.id][:,0], errors_nonconform_per_camera[camera.id][:,1], s=0.5, c='red', marker = 'o', alpha=1)  

        
        # for errors_ in errors_conform: 
        #     ax.scatter(errors_[0], errors_[1], s=0.5, c='blue', marker = 'o', alpha=1)  

        # for errors_ in errors_nonconform: 
        #     ax.scatter(errors_[0], errors_[1], s=0.5, c='red', marker = 'o', alpha=1)  
      
        circle = plt.Circle((0,0), 1, edgecolor='red', facecolor='none', linewidth=1.5, alpha = 1)
        ax.add_patch(circle)

        circle = plt.Circle((0,0), 0.7, edgecolor='red', facecolor='none', linewidth=1.5, alpha = 1)
        ax.add_patch(circle)


    
    # axIndex = 0
    for camera in scene_estimate.cameras.values(): 
        # ax = axs[axIndex]
        # axIndex += 1
        if save_fig:
            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show(block=False)
            



    return 