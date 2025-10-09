# Set the DISPLAY environment variable to point to the Windows host
export DISPLAY=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0
unset LIBGL_ALWAYS_INDIRECT
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
# (optional) avoid DRI3 oddities
export LIBGL_DRI3_DISABLE=1
