#!/bin/bash
####################################################################################################

export ANACONDA_ENV_NAME="radler-remote"
#   The name of the Conda environment you want the script to operate on.

export PORTABLECUDA_ROOT="$HOME/portablecuda/10.0.130.1/"
#   A directory which contains the root-level installation of:
#   -> CUDA Toolkit for Linux, v. 10.0.130
#   -> CUDNN for Linux, v. 7.5.1 for CUDA 10.0
#   -> NCCL for Linux, v. 2.4.8 for CUDA 10.0
#   and the /lib64/ directory renamed to (and merged with) /lib/.

export ANACONDA_BASEDIR_NAME="anaconda3"
#   The name of the (in-home, i.e. ~/) directory in which [Ana|Mini]conda is installed.

####################################################################################################

# Copy and link CUDA
cp -R -np $PORTABLECUDA_ROOT/* "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME"
ln -s "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/lib" "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/lib64"

# Link CUDNN to another common place (for Theano and the like)
mkdir -p "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/x86_64-conda_cos6-linux-gnu/sysroot/lib"
ln -s "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/lib/libcudnn.so" "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/x86_64-conda_cos6-linux-gnu/sysroot/lib/libcudnn.so"

# Prepare Anaconda environment setup (if necessary)
sed -i "s/PLEASE_REPLACE_ME_1/$ANACONDA_ENV_NAME/g" ./etc/conda/activate.d/zzz_cuda.sh
sed -i "s/PLEASE_REPLACE_ME_2/$ANACONDA_BASEDIR_NAME/g" ./etc/conda/activate.d/zzz_cuda.sh

# Copy Anaconda environment setup
cp -R -f ./etc "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME"
