#!/bin/bash

# Configure display for WSL2
export DISPLAY=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0
unset LIBGL_ALWAYS_INDIRECT
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export LIBGL_DRI3_DISABLE=1

# Activate habitat environment and run the interactive tool
source ~/anaconda3/etc/profile.d/conda.sh
conda activate habitat
python -m src.main
