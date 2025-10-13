# physical-ai-hw1

NYCU Physical AI 2025 Fall

Spec: [Google Docs](https://docs.google.com/document/d/1UqDRjh7qwQVzz2iN9Abdu4-NIl-xEO4G/edit)

## Preparation
The replica dataset, you can use the same one in `hw0`.

#### if using WSL2 on Windows

Install VcXsrv on your device by [official website](https://vcxsrv.com/)

Configure your VcXsrv to allow connections from WSL2. You can do this by running the following command in your WSL2 terminal:

```bash
# Set the DISPLAY environment variable to point to the Windows host
export DISPLAY=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0
unset LIBGL_ALWAYS_INDIRECT
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
# (optional) avoid DRI3 oddities
export LIBGL_DRI3_DISABLE=1
```

Open VcXsrv with the following configuration:

1. At first page, please check the following options and set the display number:
   - [x] Choose Multiple windows
   - Set display number to -1
2. At second page, please check the following options:
   - [x] Start no client
     - [x] Clipboard: Primary Selection
   - [x] Native opengl
   - [x] Disable access control
3. Optional: Save the configuration file for later use
4. Now, you may see the X server icon on your taskbar (Taskbar is at the down-right corner of your screen with `^` icon)

## Reproduce the results

### 1. Bird's Eye View (BEV) Projection

```bash
# Activate conda environment
conda activate habitat

# Run BEV to front view projection
python bev.py
```

This script allows you to:
- Select pixels on a bird's-eye view (top-down) image by clicking
- Project the selected region to a front view (perspective) image
- The script processes two image pairs by default:
  - `bev_data/bev1.png` → `bev_data/front1.png`
  - `bev_data/bev2.png` → `bev_data/front2.png`
- Output files are saved in the `output/` directory:
  - `selected_pixels_front1.png` and `selected_pixels_front2.png` (marked BEV images)
  - `projection_front1.png` and `projection_front2.png` (projected front view)

**Usage Instructions:**
1. A window will pop up showing the BEV image
2. Click on the image to select points (left mouse button)
3. Close the window when done selecting points
4. The projected region will be highlighted on the front view image
5. Close the front view window after viewing the result
6. Continue to the next image pair

### 2. Collect the simulation data

```bash
# Activate conda environment
conda activate habitat

# Collect Floor 1 dataset
python load.py -f 1

# Collect Floor 2 dataset
python load.py -f 2
```

**Usage Instructions:**
1. The script will start a Habitat simulation environment
2. Three windows will appear showing RGB, depth, and semantic sensor views
3. Use keyboard controls to navigate the agent:
   - Press `w` to move forward
   - Press `a` to turn left
   - Press `d` to turn right
   - Press `f` to finish and save the collected data
4. The script will save RGB, depth, and semantic images at each step
5. Camera poses (position and rotation) are automatically recorded
6. All data is saved in the `data_collection/` directory:
   - `first_floor/` or `second_floor/` depending on the floor selected
   - Each folder contains `rgb/`, `depth/`, `semantic/` subdirectories
   - Ground truth poses are saved as `GT_pose.npy`

### 3. Reconstruction and Trajectory Estimation

```bash
# Activate conda environment
conda activate habitat

# Reproduce Floor 1 results with manually implemented ICP
python reconstruct.py

# Reproduce Floor 1 results with Open3D ICP
python reconstruct.py --version open3d

# Reproduce Floor 2 results with manually implemented ICP
python reconstruct.py -f 2

# Reproduce Floor 2 results with Open3D ICP
python reconstruct.py -f 2 --version open3d
```

**Usage Instructions:**
1. The script will process the collected RGB and depth images from the specified floor
2. It performs the following steps automatically:
   - Converts depth images to 3D point clouds
   - Down-samples point clouds using voxel down-sampling
   - Registers consecutive frames using ICP (Iterative Closest Point) algorithm
   - Estimates camera trajectory by accumulating transformations
3. Two ICP implementations are available:
   - `my_icp` (default): Manually implemented ICP algorithm
   - `open3d`: Uses Open3D's built-in ICP implementation
4. After processing, the script will:
   - Display the mean L2 distance between estimated and ground truth trajectories
   - Open a 3D visualization window showing:
     - Reconstructed point cloud (colored)
     - Estimated trajectory (red line)
     - Ground truth trajectory (black line)
5. Close the visualization window when finished viewing the results
6. Note: Ceiling points are automatically removed for better visualization
