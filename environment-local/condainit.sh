#!/bin/bash
####################################################################################################

export ANACONDA_ENV_NAME="radler-local"
#   The name of the Conda environment you want the script to operate on.

export ANACONDA_BASEDIR_NAME="anaconda3"
#   The name of the (in-home, i.e. ~/) directory in which [Ana|Mini]conda is installed.

export LIBTORCH_ROOT="/opt/libtorch"
#   The path to the directory in which LibTorch resides.

####################################################################################################

# Become location-aware
export SELF_STORED_CALLDIR="$(pwd)"

# Remove already-existing environment with the same name
conda env remove -n $ANACONDA_ENV_NAME

# Create environment
# NOTE: Specific setups may break if the PYTHONUSERBASE is set before this script is called, without the workaround below!
export PRE_PYTHONUSERBASE="$PYTHONUSERBASE"
export PYTHONUSERBASE="$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME"
conda env create -f environment.yml
export PYTHONUSERBASE="$PRE_PYTHONUSERBASE"
unset PRE_PYTHONUSERBASE

# Remove Anaconda-installed CUDA & other nasty stuff (that must be system-installed, though); install and overwrite (if any) libjpeg-turbo
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/activate" $ANACONDA_ENV_NAME
conda remove -y cmake curl krb5 mpi cudatoolkit cudnn nccl nccl2 --force
conda install -y libjpeg-turbo --force --no-deps
#   Cmake      -> use upstream: better module-detection rules;
#   Curl       -> use upstream: better support for SSL
#   Kerberos   -> use upstream: integrated version fails on rolling distributions
#   MPI        -> use custom-compiled: better integration with CUDA/Hwloc
#   CUDA & co. -> use script 'cudacopy.sh' (see below)
#   LibJPEG    -> libjpeg-turbo is a drop-in replacement of libjpeg that may be overwritten by libjpeg during install

rm -f "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/compiler_compat/ld"
ln -s "$(which ld)" "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/compiler_compat/"
#   Replaced because it fails with very recent GlibC/LibC(++), as in rolling distros

source "$HOME/$ANACONDA_BASEDIR_NAME/bin/deactivate"

# Apply the `cudacopy` patches and install the LibTorch libraries (C++)
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/activate" $ANACONDA_ENV_NAME
# cudacopy
./cudacopy.sh
# LibTorch
## cp -R -np $LIBTORCH_ROOT/* "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME"
# Deactivate environment
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/deactivate"

# Install: (0) TorchX; (1) PyTorch Lightning; (2) Pillow-SIMD; (3) Hy; (4) CuPy; (5) APEX; (6) DALI weekly; (7) Horovod; (8) Hydra; (9) Flax; (10) Optimization-related stuff.
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/activate" $ANACONDA_ENV_NAME
pip install git+https://github.com/SurrealAI/torchx-public.git
pip install --no-cache-dir --upgrade --no-deps --force-reinstall git+https://github.com/williamFalcon/pytorch-lightning.git
CC="gcc -mavx2" pip install --no-cache-dir --upgrade --no-deps --force-reinstall --no-binary :all: --compile pillow-simd
#CC="gcc -mavx2" pip install --no-cache-dir --upgrade --no-deps --force-reinstall --no-binary :all: --compile https://github.com/uploadcare/pillow-simd.git
pip install git+https://github.com/hylang/hy.git
pip install --upgrade --no-deps --pre cupy-cuda101
#
pip install --upgrade --no-deps git+https://github.com/google/flax.git
pip install --upgrade git+https://github.com/adamhaber/JaxEnt.git
pip install --upgrade --no-deps --force learn2learn
pip install --upgrade --no-deps --force cherry-rl
pip install --upgrade --no-deps --force torchmeta
#
# TensorFlow stuff
pip install "dm-sonnet>=2.0" --pre
#
# Optimization-related stuff
MARCH_NATIVE=1 OPENMP_FLAG="-fopenmp" pip install diffcp
pip install git+https://github.com/cvxgrp/cvxpylayers.git
pip install git+https://github.com/cvxgrp/cvxpyrepair.git
#
#
# Chainer remnants & co.
pip install --upgrade chainer --pre
pip install --upgrade git+https://github.com/cybertronai/pytorch-sso.git
#
#
# WESSELB
pip install git+https://github.com/wesselb/matrix.git git+https://github.com/wesselb/wbml.git git+https://github.com/wesselb/stheno.git git+https://github.com/wesselb/varz.git
pip install git+https://github.com/wesselb/matrix.git git+https://github.com/wesselb/wbml.git git+https://github.com/wesselb/stheno.git git+https://github.com/wesselb/varz.git
#
# Jupinx/QuantEcon stack
##
pip install --no-deps sphinxcontrib-jupyter
pip install --no-deps jupinx
#
jupyter nbextension enable varInspector/main
jupyter labextension install @jupyterlab/toc --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install plotlywidget --no-build
jupyter labextension install jupyterlab-plotly --no-build
jupyter labextension install @lckr/jupyterlab_variableinspector
jupyter labextension install @krassowski/jupyterlab-lsp
jupyter labextension install @jupyterlab/debugger
jupyter labextension install @pyviz/jupyterlab_pyviz
#
jupyter lab build
##
#
pip install --upgrade jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
jupyter serverextension enable nteract_on_jupyter
jupyter serverextension enable elyra
#
#
jupyter lab build
#
# FEniCS
#
#git clone https://bitbucket.org/fenics-project/dolfin
#git clone https://bitbucket.org/fenics-project/mshr
#git clone https://github.com/FEniCS/dolfinx.git
#mkdir dolfin/build
#cd dolfin/build
#ccmake ../
#sudo make install -j12
#cd ../..
#mkdir mshr/build
#cd mshr/build
#ccmake ../
#sudo make install -j12
#cd ../..
#cd dolfin/python
#pip3 install . --upgrade --no-deps
#cd ../..
#cd mshr/python
#pip3 install . --upgrade --no-deps
#cd ../..
##
#bash -c "read -p 'Ready to continue...? [ENTER]'"
#
##
##
# NOTE: moved down; just look after the gcc-7 trick (1st of the two blocks).
#pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
#pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly/cuda/10.1 nvidia-dali-weekly
CC=gcc-7 CXX=g++-7 FC=gfortran-7 HOROVOD_NCCL_HOME="$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME" HOROVOD_NCCL_INCLUDE="$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/include" HOROVOD_NCCL_LIB="$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/lib" HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
pip install hydra-core --pre
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/deactivate"


# Install Nvidia APEX (with cpp & cuda extensions) with a bold, unorthodox trick
################################################################################
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/activate" $ANACONDA_ENV_NAME

export PTG_PREPATH="$PATH"
export PTG_FAKEPATHDIR="$(pwd)"

mkdir -p ./faketop && cd ./faketop

export PATH="$PTG_FAKEPATHDIR/faketop:$PATH"

ln -s $(which gcc-7) ./gcc
ln -s $(which g++-7) ./g++
ln -s $(which gfortran-7) ./gfortran

cd ..

# APEX (from above)
pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git

export PATH="$PTG_PREPATH"
unset PTG_PREPATH
rm -R -f "$PTG_FAKEPATHDIR/faketop"
unset PTG_FAKEPATHDIR

source "$HOME/$ANACONDA_BASEDIR_NAME/bin/deactivate"
################################################################################
# Phew! Done! :)


# Install the entire 'PyTorch Geometric' stack with a bold, unorthodox trick (again!)
######################################################################################
source "$HOME/$ANACONDA_BASEDIR_NAME/bin/activate" $ANACONDA_ENV_NAME

export PTG_PREPATH="$PATH"
export PTG_FAKEPATHDIR="$(pwd)"

mkdir -p ./faketop && cd ./faketop

export PATH="$PTG_FAKEPATHDIR/faketop:$PATH"

ln -s $(which gcc-5) ./gcc
ln -s $(which g++-5) ./g++
ln -s $(which gfortran-5) ./gfortran

cd ..

pip install https://pytorch-geometric.com/whl/torch-1.5.0/torch_scatter-latest%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install https://pytorch-geometric.com/whl/torch-1.5.0/torch_sparse-latest%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install https://pytorch-geometric.com/whl/torch-1.5.0/torch_cluster-latest%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install https://pytorch-geometric.com/whl/torch-1.5.0/torch_spline_conv-latest%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install git+https://github.com/rusty1s/pytorch_geometric.git

export PATH="$PTG_PREPATH"
unset PTG_PREPATH
rm -R -f "$PTG_FAKEPATHDIR/faketop"
unset PTG_FAKEPATHDIR

source "$HOME/$ANACONDA_BASEDIR_NAME/bin/deactivate"
######################################################################################
# Phew! Done! :)

# Install manually just some more packages...
cd "$HOME/$ANACONDA_BASEDIR_NAME/envs/$ANACONDA_ENV_NAME/lib/python3.7/site-packages/"
git clone --branch master --depth 1 --recursive https://github.com/emaballarin/hcgd.git
#git clone --branch master --depth 1 --recursive https://github.com/emaballarin/RAdam.git
git clone --branch master --depth 1 --recursive https://github.com/emaballarin/lookahead_pytorch.git
git clone --branch master --depth 1 --recursive https://github.com/emaballarin/l4_pytorch.git
git clone --branch master --depth 1 --recursive https://github.com/emaballarin/rangeropt.git
git clone --branch master --depth 1 --recursive https://github.com/MilesCranmer/lagrangian_nns.git

# End
cd "$SELF_STORED_CALLDIR"
echo ' '
echo 'DONE!'
