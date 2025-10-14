# physical-ai-hw2
NYCU Physical AI 2025 Fall

Spec: https://drive.google.com/file/d/1jg5wRDpTQcx7Ux01hNzPmMdKGN-Mhxc0/view?usp=sharing
## Preparation
In your original physical-ai25 directory, `git pull` to get new `hw2` directory.

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

### 2. Download dataset

Download dataset from [here](https://drive.google.com/file/d/1zHA2AYRtJOmlRaHNuXOvC_OaVxHe56M4/view)
and put the directory under `replica_v1/`

After downloading and unzipping, your directory structure should look like this:
```
hw0/
├── habitat-lab/
├── habitat-sim/
├── replica_v1/
│   ├── __MACOSX/
│   ├── apartment_0/
│   │   ├── habitat/
│   │   ├── textures/
│   │   └── <Other files>
│   └── .gitignore
├── load.py
├── README.md
└── requirements.txt
```

### Reproduce the result

```bash
# Activate conda environment
conda activate habitat

# Run the code (Note: Must run as module)
python -m src.main
```
