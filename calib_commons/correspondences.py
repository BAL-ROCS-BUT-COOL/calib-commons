from typing import Dict, List

# from externalCalibrationPlanarPoints.calib.observation import Observation

# from externalCalibrationPlanarPoints.calib.types import idtype

from calib_commons.types import idtype
from calib_commons.observation import Observation

Correspondences = Dict[idtype, Dict[idtype, Observation]]

def get_conform_obs_of_cam(cam_id: idtype, 
                             correspondences: Correspondences) -> Dict[idtype, Observation]: 
    return {id: obs for id, obs in correspondences[cam_id].items() if obs._is_conform}


def get_tracks_of_points(point_ids: List[idtype], correspondences: Correspondences) -> Dict[idtype, set[idtype]]:
    tracks = {}
    for point_id in point_ids:
        tracks[point_id] = set()
        for cam_id, observations in correspondences.items():
            if point_id in observations and observations[point_id]._is_conform:
                tracks[point_id].add(cam_id)
    return tracks

def get_tracks_length_of_points(point_ids: List[idtype], correspondences: Correspondences) -> Dict[idtype, int]:
    tracks = get_tracks_of_points(point_ids, correspondences)
    return {point_id: len(track) for point_id, track in tracks.items()}


def get_tracks(correspondences) -> Dict[idtype, set[idtype]]: 
    tracks = {}
    for camera_id, observations in correspondences.items():
        # print("")
        # print(f"cam {camera_id}")
        for point_id, observation in observations.items():
            # print(f"pt {point_id}")

            if not (point_id in tracks):
                # print("new set")
                tracks[point_id] = set()
            if observation._is_conform:
                tracks[point_id].add(camera_id)
            
            # print(tracks)
            # print("")
                    
    return tracks
    

# def filter_with_track_length(correspondences, points_ids, min_track_length) -> set[idtype]: 
#     tracks = get_tracks(correspondences)
#     points_with_long_enough_track = set([checker_id for checker_id, track in tracks.items() if len(track) >= min_track_length])
#     valid_points_ids = points_ids & checkers_with_long_enough_track
#     return valid_points_ids

def filter_correspondences_with_track_length(correspondences: Correspondences, 
                                             min_track_length: int) -> Correspondences: 
    tracks = get_tracks(correspondences)
    points_with_long_enough_track = set([point_id for point_id, track in tracks.items() if len(track) >= min_track_length])

    correspondences_filtered = {}
    for camera_id, observations in correspondences.items():
        correspondences_filtered[camera_id] = {}
        for point_id, observation in observations.items():
            if point_id in points_with_long_enough_track:
                correspondences_filtered[camera_id][point_id] = observation
    return correspondences_filtered

# def filter_with_min_max_id(correspondences: Correspondences, 
#                            id_min: idtype, 
#                            id_max: idtype) -> Correspondences:
    
#     correspondences_filtered = {}
#     for camera_id, observations in correspondences.items():
#         correspondences_filtered[camera_id] = {id: obs for id, obs in observations.items() if (not id_min or id >= id_min) and (not id_max or id <= id_max)}
#     return correspondences_filtered
