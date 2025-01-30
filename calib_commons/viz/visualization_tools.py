from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation 
from matplotlib import rcParams


COLORS = {
    1: "#FF5733",  # Red
    2: "#33FF57",  # Lime Green
    3: "#3357FF",  # Blue
    4: "#F1C40F",  # Yellow
    5: "#8E44AD",  # Purple
    6: "#3498DB",  # Light Blue
    7: "#E67E22",  # Orange
    8: "#2ECC71",  # Mint
    9: "#1ABC9C",  # Teal
    10: "#9B59B6",  # Amethyst
    11: "#34495E",  # Navy Blue
    12: "#16A085",  # Sea Green
    13: "#27AE60",  # Nephritis
    14: "#2980B9",  # Belize Hole Blue
    15: "#8E44AD",  # Wisteria Purple
    16: "#F39C12",  # Sunflower Yellow
    17: "#D35400",  # Pumpkin Orange
    18: "#C0392B",  # Pomegranate Red
    19: "#BDC3C7",  # Silver
    20: "#7F8C8D",  # Asbestos
    21: "#99A3A4",  # Concrete
    22: "#7D3C98",  # Plum
    23: "#2874A6",  # Blue Sapphire
    24: "#A569BD",  # Lavender
    25: "#D2B4DE",  # Pastel Purple
    26: "#AED6F1",  # Light Blue
    27: "#F5B041",  # Tiger's Eye
    28: "#DC7633",  # Cinnamon
    29: "#AEB6BF",  # Cool Grey
    30: "#1B4F72"   # Dark Blue
}

IDMIN = 0
IDMAX = 0

def set_color_map(checker_ids):
    if checker_ids:
        global IDMIN, IDMAX
        IDMIN = min(checker_ids)
        IDMAX = max(checker_ids)


def get_color_from_id(id, cmap_="random"): 
    # id = int(id.split('_')[1])
    if cmap_ == "random": 
        id = int(id) % 30 + 1
        color = COLORS[id]
    else:
        cmap = plt.get_cmap('coolwarm')
        if IDMIN == IDMAX: 
            return 'blue'
        i = (id - IDMIN) / (IDMAX - IDMIN)
        color = cmap(i)
    return color

def get_color_from_error(error, error_min, error_max): 
    cmap = plt.get_cmap('coolwarm')
    i = (error - error_min) / (error_max - error_min)
    color = cmap(i)
    return color

def create_color_list(N, start_color, end_color):
    cmap = plt.get_cmap('coolwarm')  # You can change 'coolwarm' to any other colormap like 'viridis', 'plasma', etc.
    colors = [cmap(i / (N - 1)) for i in range(N)]
    return colors

def plot_camera_pyramid(extrinsic, color='r', focal_len_scaled=5, aspect_ratio=1, aspect_ratio_x_y=1, ax=None):
    if ax is None: 
        ax = plt.gca()

    vertex_std = np.array([[0, 0, 0, 1],
                           [focal_len_scaled * aspect_ratio*aspect_ratio_x_y, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                           [focal_len_scaled * aspect_ratio*aspect_ratio_x_y, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio*aspect_ratio_x_y, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                           [-focal_len_scaled * aspect_ratio*aspect_ratio_x_y, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
    vertex_transformed = vertex_std @ extrinsic.T
    meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
              [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
              [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
              [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
              [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
    pyramid = Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
    ax.add_collection3d(pyramid)

def plot_se3_frame(T, axis_length=0.1, ax=None):
    if ax is None:
        ax = plt.gca()

    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([axis_length, 0, 0, 1])
    y_axis = np.array([0, axis_length, 0, 1])
    z_axis = np.array([0, 0, axis_length, 1])
    
    transformed_origin = T.dot(origin)
    transformed_x_axis = T.dot(x_axis)
    transformed_y_axis = T.dot(y_axis)
    transformed_z_axis = T.dot(z_axis)
    
    ax.quiver(transformed_origin[0], transformed_origin[1], transformed_origin[2],
              transformed_x_axis[0] - transformed_origin[0], transformed_x_axis[1] - transformed_origin[1], transformed_x_axis[2] - transformed_origin[2], color='r', label='X-axis')
    ax.quiver(transformed_origin[0], transformed_origin[1], transformed_origin[2],
              transformed_y_axis[0] - transformed_origin[0], transformed_y_axis[1] - transformed_origin[1], transformed_y_axis[2] - transformed_origin[2], color='g', label='Y-axis')
    ax.quiver(transformed_origin[0], transformed_origin[1], transformed_origin[2],
              transformed_z_axis[0] - transformed_origin[0], transformed_z_axis[1] - transformed_origin[1], transformed_z_axis[2] - transformed_origin[2], color='b', label='Z-axis')
    
def plot_frame(T, name, axis_length, ax=None): 
    if ax is None: 
        ax = plt.gca()

    plot_se3_frame(T, axis_length, ax)
    s = 0.15
    p = T[:3, 3] - s * T[:3, 2]
    x, y, z = p[0], p[1], p[2]
    ax.text(x, y, z, name, color='black', fontsize=10)

def plot_camera(pose, name, size=0.3, ax=None, aspect_ratio_x_y_=1): 
    plot_camera_pyramid(pose, 'c', size, 0.4, aspect_ratio_x_y_, ax)
    plot_frame(pose, name, axis_length=size, ax=ax)

if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 6))
       
    rcParams['text.usetex'] = True

    ax = fig.add_subplot(projection='3d')
     
    xlim = [-1, 1]
    ylim = xlim
    zlim = ylim
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')   
    ax.set_box_aspect([1, 1, 1]) 
    T_W_C = np.eye(4)
    T_W_C[:3, 3] = [0.5, 0.6, 0.3]
    R = Rotation.from_euler('zyx', [90, 0, 0], degrees=True).as_matrix()
    T_W_C[:3, :3] = R

    i = 3
    name = r"$\{C_" + str(i) + r"\}$"
    plot_camera(T_W_C, name, ax=ax)

    plt.show()
