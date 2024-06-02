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
git clone https://github.com/cdfan0627/3DCV_final.git --recursive
cd 3DCV_final

conda create -n DS3DGS python=3.8
conda activate DS3DGS

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```
## Render & Evaluation
```shell
python render.py -m output/NeRF-DS/as_novel_view --mode render --iteration 24000 --skip_train
python render.py -m output/NeRF-DS/basin_novel_view --mode render --iteration 24000 --skip_train
python render.py -m output/NeRF-DS/bell_novel_view --mode render --iteration 24000 --skip_train
python render.py -m output/NeRF-DS/cup_novel_view --mode render --iteration 24000 --skip_train
python render.py -m output/NeRF-DS/plate_novel_view --mode render --iteration 24000 --skip_train
python render.py -m output/NeRF-DS/press_novel_view --mode render --iteration 24000 --skip_train
python render.py -m output/NeRF-DS/sieve_novel_view --mode render --iteration 24000 --skip_train
python metrics.py --model_path "output/NeRF-DS/as_novel_view/"  
python metrics.py --model_path "output/NeRF-DS/basin_novel_view/"  
python metrics.py --model_path "output/NeRF-DS/bell_novel_view/"  
python metrics.py --model_path "output/NeRF-DS/cup_novel_view/"  
python metrics.py --model_path "output/NeRF-DS/plate_novel_view/"  
python metrics.py --model_path "output/NeRF-DS/press_novel_view/"  
python metrics.py --model_path "output/NeRF-DS/sieve_novel_view/" 
```

