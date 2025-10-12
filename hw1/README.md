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

1. Collect the simulation data

```bash
# Activate conda environment
conda activate habitat

# Collect Floor 1 dataset
python load.py -f 1

# Collect Floor 2 dataset
python load.py -f 2
```

2. Reproduce the results

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
