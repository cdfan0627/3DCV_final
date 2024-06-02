# 3DCV_final
## Dataset
We use real-world dataset from [NeRF-DS](https://github.com/JokerYan/NeRF-DS/releases/tag/v0.1-pre-release)

We organize the datasets as follows:
```shell
├── data
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── bell
│     ├── cup
│     ├── plate
│     ├── press
│     ├── sieve
```

## Environment Setup
```shell
git clone https://github.com/ingra14m/Deformable-3D-Gaussians --recursive
cd 

conda create -n DS3DGS python=3.8
conda activate DS3DGS

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```
