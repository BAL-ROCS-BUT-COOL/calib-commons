import numpy as np
def generateCircularCameras(pointToLookAt, d, tilt, nCams):
    # Extrinsics
    R_c2w = np.array([[1, 0, 0],
                      [0, np.cos(tilt + np.pi/2), -np.sin(tilt + np.pi/2)],
                      [0, np.sin(tilt + np.pi/2), np.cos(tilt + np.pi/2)]])
    
    # Distance camera center to origin
    center = np.array([[0], [-d * np.cos(tilt)], [d * np.sin(tilt)]])
    
    # Circular motion
    slant = 2 * np.pi / nCams
    poses_c2w = []
    
    for ii in range(nCams):
        slant_ii = slant * ii
        rotZ = np.array([[np.cos(slant_ii), -np.sin(slant_ii), 0],
                         [np.sin(slant_ii), np.cos(slant_ii), 0],
                         [0, 0, 1]])
        
        R_c2w_rot = np.dot(rotZ, R_c2w.T).T
        center_rot = np.dot(rotZ, center)
        center_rot_trans = center_rot + pointToLookAt[:,None]
        poses_c2w.append(np.vstack((np.hstack((R_c2w_rot.T, center_rot_trans)), [0, 0, 0, 1])))
    
    return poses_c2w


def generateCircularCameras_random(pointToLookAt, d, tilt, nCams):
    # Extrinsics
    R_c2w = np.array([[1, 0, 0],
                      [0, np.cos(tilt + np.pi/2), -np.sin(tilt + np.pi/2)],
                      [0, np.sin(tilt + np.pi/2), np.cos(tilt + np.pi/2)]])
    
    # Distance camera center to origin
    center = np.array([[0], [-d * np.cos(tilt)], [d * np.sin(tilt)]])

    # print(d * np.cos(tilt))
    # Circular motion
    # slant = 2 * np.pi / nCams
    poses_c2w = []

    min_angle = 45/180*np.pi
    # total_angle = nCams * min_angle
    # if total_angle > 2 * np.pi:
    #     raise ValueError("L'angle minimum est trop grand pour le nombre de points souhaité.")
    
    # Créer des angles espacés uniformément
    # angles = np.linspace(0, 2 * np.pi, nCams, endpoint=False)
    
    # Ajouter de petites perturbations aléatoires aux angles tout en respectant l'écart min_angle
    # for i in range(1, n):
    
    angles = []
    # used_angles = []
    for ii in range(nCams):

        while True:
            phi = np.random.uniform(0, 2 * np.pi)

            if all(abs(phi - p) >= min_angle for p in angles) and \
            all(abs(phi - p) <= 2 * np.pi - min_angle for p in angles):  # handle wrap-around at 2pi
                break

        angles.append(phi)
        slant_ii = phi
        

        # max_deviation = min((angles[ii] - angles[ii-1]) - min_angle, min_angle)
        # if max_deviation > 0:
        #     deviation = np.random.uniform(-max_deviation, max_deviation)
        #     angles[ii] += deviation

        # slant_ii = angles[ii]
        # slant_ii = slant * ii
        # slant_ii = np.random.uniform(0, 2 * np.pi)



        # used_angles.append(slant_ii)
        rotZ = np.array([[np.cos(slant_ii), -np.sin(slant_ii), 0],
                         [np.sin(slant_ii), np.cos(slant_ii), 0],
                         [0, 0, 1]])
        
        R_c2w_rot = np.dot(rotZ, R_c2w.T).T
        center_rot = np.dot(rotZ, center)
        center_rot_trans = center_rot + pointToLookAt[:,None]
        poses_c2w.append(np.vstack((np.hstack((R_c2w_rot.T, center_rot_trans)), [0, 0, 0, 1])))
    
    return poses_c2w