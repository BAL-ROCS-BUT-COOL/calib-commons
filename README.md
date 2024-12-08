# **calib-commons**

A Python package of utilities for the Python packages [`calib-board`](https://github.com/tflueckiger/calib-board) and [`calib-proj`](https://github.com/tflueckiger/calib-proj) which perform **external calibration** of multi-camera systems, and some additional tools.

---

## **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/tflueckiger/calib-commons.git
   cd calib-commons
   pip install .
   ```

2. Optional: Install in editable mode:

   ```bash
   pip install -e . --config-settings editable_mode=strict
   ```

> Dependencies: if dependencies listed in requirements.txt are not satisfied, they will be automatically installed from PyPi.

---

## **Package content**

The package contains utilities for the Python packages [`calib-board`](https://github.com/tflueckiger/calib-board) and [`calib-proj`](https://github.com/tflueckiger/calib-proj). 

In addition, the packages provides some tools: 
- [calibrate-intrinsics](#calibrate-intrinsics): CLI tool for internal calibration
- [calib-eval](#calib-eval): script for evaluating a given camera calibration using an evaluation dataset


## **calibrate-intrinsics**
The package provide a command-line tool **calibrate-intrinsics** for automatic internal calibration of multiple cameras using either videos recordings or images of a moving chessboard.

### **Option 1: Using Video Recordings**

1. Place video recordings of a moving chessboard in a common directory:

   ```plaintext
   data_directory/
   ├── camera1.mp4
   ├── camera2.mp4
   └── ...
   ```

2. Navigate to the directory:

   ```bash
   cd path_to_data_directory
   ```

3. Run the calibration command:

   ```bash
   calibrate-intrinsics --use_videos --square_size 0.03 --chessboard_width 11 --chessboard_height 8 --sampling_step 45
   ```


### **Option 2: Using Images**

1. Organize images of a moving chessboard into subfolders, one per camera:

   ```plaintext
   data_directory/
   ├── camera1/
   │   ├── frame1.jpg
   │   ├── frame2.jpg
   │   └── ...
   ├── camera2/
   │   ├── frame1.jpg
   │   ├── frame2.jpg
   │   └── ...
   └── cameraN/
       ├── frame1.jpg
       ├── frame2.jpg
       └── ...
   ```

2. Navigate to the directory:

   ```bash
   cd path_to_data_directory
   ```

3. Execute the calibration command:

   ```bash
   calibrate-intrinsics --square_size 0.03 --chessboard_width 11 --chessboard_height 8
   ```

> Note that in this case the recordings are independent (no temporal synchronization required) and that the naming of the frames is free.

> For detailed usage, run `calibrate-intrinsics --help`.

### Output 
The tool will create a result directory at the specified location (see `calibrate-intrinsics --help`), with a folder containing camera intrinsics in .json files. 
```plaintext
camera_intrinsics/
├── camera1_intrinsics.json
├── camera2_intrinsics.json
└── ...
```

---
## **calib-eval**
The package provide a script tool to evaluate a given calibration (= intrinsics + extrinsics) of a multi-camera systems obtained via any calibration method. 

The **inputs** of the tool are: 
- **1. Calibration parameters**: intrinsics + extrinsics
- **2. Evaluation dataset**: synchronized images of a moving calibration pattern (chessboard or ChArUco board)

### **Calibration parameters**


#### Intrinsics 
The intrinsics are specified via the path to a folder containing intriniscs in .json files. 
```plaintext
camera_intrinsics/
├── camera1_intrinsics.json
├── camera2_intrinsics.json
└── ...
```
> Intrinsics can be obtained using [calibrate-intrinsics](#calibrate-intrinsics). In this case, the output folder of [calibrate-intrinsics](#calibrate-intrinsics) matches the required format directly.


#### Extrinsics 
The extrinsics are specified via a .json file containing the pose (translation and ZYX euler angles) of each camera.

> Extrinsics can be obtained using either [`calib-board`](https://github.com/tflueckiger/calib-board) or [`calib-proj`](https://github.com/tflueckiger/calib-proj). In both cases, they output a camera_poses.json file matching the required format.

### **Evaluation dataset**

> Evaluation dataset: it is essential that the dataset used for evaluation is separate and independent from the calibration dataset to ensure an accurate and unbiased evaluation of the calibration method.

The evaluation dataset consist of synchronized images of a moving calibration pattern (chessboard or ChArUco board). The images must fit the following structure: 


```plaintext
images_directory/
├── camera1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── camera2/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── cameraN/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

- Each subdirectory (`camera1`, `camera2`, ..., `cameraN`) contains images for a single camera.
- Filenames **must match across cameras** to ensure synchronization (e.g., `1.jpg` in `camera1` corresponds to `1.jpg` in `camera2`).

### How does the evaluation works?
Individual correspondences between cameras are obtained using the synchronized images of the calibration board. *Individual* correspondences means that membership to boards are not considered. The calibration board is only used to obtain accurate inter-camera correspondences. The evaluation consist of **inimizing the total reprojection error**:

- First, initial estimates of 3D points are obtained by performing pairwise triangulation between cameras. If a 3D point is calculated multiple times from different camera pairs, the median of these computed coordinates is taken as the initial estimate for that 3D point.

- Subsequently, the 3D points are refined by minimizing the total reprojection error, **keeping the camera poses fixed during this optimization process**.

### Usage

1. Edit the user interface parameters in `scripts/run_calib_eval.py`:
   - **Board Detection Parameters**: Configure settings for detecting the calibration pattern.
   - **Evaluation Parameters**: reprojection error threshold

> The reprojection error threshold is used in the iterative filtering of the evaluation algorithm to discard observations that have a corresponding reprojection error higher than the specified threshold. 

2. Run the script:

   ```bash
   python calib_commons/scripts/run_calib_eval.py
   ```
The evaluation metrics (notably the **reprojection errors**) are saved in results/metrics.json.

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/tflueckiger/calib-board/blob/main/LICENSE) file for details.

---

## **Acknowledgments**

TODO