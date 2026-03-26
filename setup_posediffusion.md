# PoseDiffusion Environment Setup Guide

## Prerequisites
- CUDA 11.7 or 11.8 driver

---

## Step 0: Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
rm /tmp/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

✅ Check:
```bash
conda --version
# Expected: conda 23.x.x or higher
```

---

## Step 1: Accept Conda Terms of Service

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## Step 2: Remove existing environment (if any)

```bash
conda deactivate
conda env remove -n posediffusion -y
```

---

## Step 3: Create conda environment

```bash
conda create -n posediffusion python=3.9 -y
conda activate posediffusion
```

✅ Check:
```bash
python --version
# Expected: Python 3.9.x
```

---

## Step 4: Install PyTorch 1.13.0 + CUDA 11.7

```bash
conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

✅ Check:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 1.13.0 / True
```

---

## Step 5: Fix MKL version

```bash
conda install mkl=2021.4.0 -c conda-forge -y
```

✅ Check:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must still be: 1.13.0 / True
```

---

## Step 6: Install fvcore + iopath

```bash
pip install fvcore iopath
```

✅ Check:
```bash
python -c "import fvcore; import iopath; print('OK')"
```

---

## Step 7: Install PyTorch3D (via pip wheel)

```bash
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html
```

✅ Check:
```bash
python -c "import pytorch3d; print(pytorch3d.__version__)"
# Expected: 0.7.5
```

⚠️ Verify torch is still OK:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must still be: 1.13.0 / True
```

---

## Step 8: Fix setuptools (required for visdom build)

```bash
pip install setuptools --upgrade
```

---

## Step 9: Install pip dependencies

```bash
pip install setuptools==69.5.1 --force-reinstall
pip install omegaconf opencv-python einops
pip install visdom --no-build-isolation
pip install accelerate==0.24.0
pip install hydra-core --upgrade
```

✅ Check:
```bash
python -c "import hydra, omegaconf, cv2, einops, visdom, accelerate; print('OK')"
```

---

## Step 10: Fix numpy + install narwhals

```bash
pip install "numpy<2" narwhals
```

✅ Check:
```bash
python -c "import numpy; print(numpy.__version__)"
# Expected: 1.26.x (NOT 2.x)
```

---

## Step 11: Clone HLoc (pycolmap 0.4.0 compatible commit)

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git dependency/hloc
cd dependency/hloc
git checkout 8eb9977
pip install -e . --no-deps
cd ../../
```

✅ Check:
```bash
python -c "import hloc; print('hloc OK')"
```

---

## Step 12: Install kornia 0.6.12 (must use --no-deps!)

```bash
pip install kornia==0.6.12 --no-deps
pip install h5py --no-deps
```

⚠️ Verify torch is still 1.13.0 after this step:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Must still be: 1.13.0 / True
```

---

## Step 13: Install pycolmap 0.4.0

```bash
pip install "pycolmap>=0.3.0,<=0.4.0"
```

✅ Check:
```bash
python -c "import pycolmap; print(pycolmap.__version__)"
# Expected: 0.4.0
```

---

## Step 14: Install other dependencies

```bash
pip install gdown matplotlib plotly h5py
pip install git+https://github.com/cvg/LightGlue --no-deps
pip install contourpy fonttools importlib-resources kiwisolver python-dateutil pyparsing cycler
```

---

## Final Verification

```bash
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
```

Expected output:
```
torch:      1.13.0
cuda:       True
pytorch3d:  0.7.5
numpy:      1.26.x
ALL OK
```

---

## Run Demo

```bash
cd pose_diffusion

# Without GGS (~4 seconds)
python demo.py image_folder="samples/apple" ckpt="../ckpts/co3d_model_Apr16.pth"

# With GGS (~150 seconds, more accurate)
python demo.py image_folder="samples/apple" ckpt="../ckpts/co3d_model_Apr16.pth" GGS.enable=true

# RealEstate10K model (must set image_size=336)
python demo.py image_folder="samples/apple" ckpt="../ckpts/re10K_img336_v2.bin" image_size=336
```
