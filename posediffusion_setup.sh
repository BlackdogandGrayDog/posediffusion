#!/usr/bin/env bash

# PoseDiffusion environment setup script
# Generated directly from the provided step-by-step guide.

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
rm /tmp/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

# Check:
conda --version
# Expected: conda 23.x.x or higher

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda deactivate
conda env remove -n posediffusion -y

conda create -n posediffusion python=3.9 -y
conda activate posediffusion

# Check:
python --version
# Expected: Python 3.9.x

conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Check:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 1.13.0 / True

conda install mkl=2021.4.0 -c conda-forge -y

# Check:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must still be: 1.13.0 / True

pip install fvcore iopath

# Check:
python -c "import fvcore; import iopath; print('OK')"

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html

# Check:
python -c "import pytorch3d; print(pytorch3d.__version__)"
# Expected: 0.7.5

# Verify torch is still OK:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must still be: 1.13.0 / True

pip install setuptools --upgrade

pip install setuptools==69.5.1 --force-reinstall
pip install omegaconf opencv-python einops
pip install visdom --no-build-isolation
pip install accelerate==0.24.0
pip install hydra-core --upgrade

# Check:
python -c "import hydra, omegaconf, cv2, einops, visdom, accelerate; print('OK')"

pip install "numpy<2" narwhals

# Check:
python -c "import numpy; print(numpy.__version__)"
# Expected: 1.26.x (NOT 2.x)

git clone --recursive https://github.com/cvg/Hierarchical-Localization.git dependency/hloc
cd dependency/hloc
git checkout 8eb9977
pip install -e . --no-deps
cd ../../

# Check:
python -c "import hloc; print('hloc OK')"

pip install kornia==0.6.12 --no-deps
pip install h5py --no-deps

# Verify torch is still 1.13.0 after this step:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must still be: 1.13.0 / True

pip install "pycolmap>=0.3.0,<=0.4.0"

# Check:
python -c "import pycolmap; print(pycolmap.__version__)"
# Expected: 0.4.0

pip install gdown matplotlib plotly h5py
pip install git+https://github.com/cvg/LightGlue --no-deps
pip install contourpy fonttools importlib-resources kiwisolver python-dateutil pyparsing cycler

python -c "
import torch
import pytorch3d
import fvcore
import hydra
import cv2
import einops
print('torch:     ', torch.__version__)
print('cuda:      ', torch.cuda.is_available())
print('pytorch3d: ', pytorch3d.__version__)
print('numpy:     ', __import__('numpy').__version__)
print('ALL OK')
"

cd pose_diffusion

python demo.py image_folder="samples/apple" ckpt="../ckpts/co3d_model_Apr16.pth"
python demo.py image_folder="samples/apple" ckpt="../ckpts/co3d_model_Apr16.pth" GGS.enable=true
python demo.py image_folder="samples/apple" ckpt="../ckpts/re10K_img336_v2.bin" image_size=336
