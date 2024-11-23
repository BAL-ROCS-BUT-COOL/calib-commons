import numpy as np

# Function to generate a rotation matrix around the x-axis
def rotX(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, cos_theta, -sin_theta],
                     [0, sin_theta, cos_theta]])

# Function to generate a rotation matrix around the y-axis
def rotY(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([[cos_theta, 0, sin_theta],
                     [0, 1, 0],
                     [-sin_theta, 0, cos_theta]])

# Function to generate a rotation matrix around the z-axis
def rotZ(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([[cos_theta, -sin_theta, 0],
                     [sin_theta, cos_theta, 0],
                     [0, 0, 1]])

