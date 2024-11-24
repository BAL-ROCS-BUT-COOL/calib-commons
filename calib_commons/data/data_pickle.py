import pickle
# from externalCalibrationPlanarPoints.calib.correspondences import Correspondences

# Function to save data to pickle
def save_to_pickle(file_path: str, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Function to load data from pickle
def load_from_pickle(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)