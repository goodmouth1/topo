#!/bin/bash
# Activate the environment (ensure conda is available)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_gpu

# Export necessary paths for Theano/CUDA
export CPATH=$CPATH:$CONDA_PREFIX/include
export LIBRARY_PATH=$LIBRARY_PATH:$CONDA_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Run the command passed as arguments
exec "$@"
