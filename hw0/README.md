# pysical-ai-hw0

NYCU Physical AI 2025 Fall

Spec: https://docs.google.com/document/d/1AGyn_86wDJFglbIqMNx1RYWl1Sz4GJe2/edit

## Introduction 
In this course, we are going to build an indoor navigation system in Habitat step by step during homework 1 ~ 2 and the final project. This homework 0 will help you to build the environment with essential packages.

## Requirements
- OS : Ubuntu Desktop 18.04, 20.04

    - Ubuntu on virtual machine is not recommended
    - MacOS may work but not guaranteed
- Python 3.7 ( You can use conda to create new environment )

## Installation

### 1. Clone the repo
`git clone git@github.com:HCIS-Lab/physical-ai25.git` to download the repo or create a new fork on you own GitHub account.

```bash
cd physical-ai25/hw0
# Ensure the latest submodules
git submodule update --init --recursive
# Create a conda env
conda create -n habitat python=3.7
# Activate the conda env
conda activate habitat
# Install requirements
pip install -r requirements.txt
# Install habitat-sim from source
cd habitat-sim && pip install -r requirements.txt && python setup.py install --bullet && cd ..
# Install habitat-lab
cd habitat-lab && pip install -r requirements.txt && python setup.py develop && cd ..
```

#### if using WSL2 on Windows

Install VcXsrv on your device by [official website](https://vcxsrv.com/)

Configure your VcXsrv to allow connections from WSL2. You can do this by running the following command in your WSL2 terminal:

```bash
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

## Tasks

1. Run the `load.py` to explore the scene

    Use keyboard to control the agent
    ```
    w for go forward  
    a for turn left  
    d for trun right  
    f for finish and quit the program
    ```

    **Note**

    You can change agent_state.position to set the agent in the first or second floor. (0, 0, 0) for first floor and (0, 1, 0) for second floor.


2. **Record the video on screen (<30 sec)** at the same time to demonstrate that the simulation can be opened up successfully.

    
