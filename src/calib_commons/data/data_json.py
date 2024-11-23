
import json
import numpy as np

from calibCommons.utils.se3 import q_from_T
from calibCommons.camera import Camera

def save_cameras_poses_to_json(cameras, path): 
        data = {}
        for camera in cameras.values(): 
            q = q_from_T(camera.pose.mat)
            euler = q[:3] * 180 / np.pi
            t = q[3:]
            data[camera.id] = {"euler_ZYX": euler.tolist(), "t": t.tolist()}
        with open(path, 'w') as f: 
            json.dump(data, f, indent=4)


def load_cameras_poses_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data