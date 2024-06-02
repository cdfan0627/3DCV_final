# 3DCV_final
## Dataset
We use real-world dataset from [NeRF-DS](https://github.com/JokerYan/NeRF-DS/releases/download/v0.1-pre-release/NeRF-DS.dataset.zip)

We organize the datasets as follows:
```shell
├── data
│   | NeRF-DS
│     ├── as_novel_view
│     ├── basin_novel_view
│     ├── bell_novel_view
│     ├── cup_novel_view
│     ├── plate_novel_view
│     ├── press_novel_view
│     ├── sieve_novel_view
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
## Weight
Download 

We organize the weights as follows:
```shell
├── output
│   | NeRF-DS
│     ├── as_novel_view
│     ├── basin_novel_view
│     ├── bell_novel_view
│     ├── cup_novel_view
│     ├── plate_novel_view
│     ├── press_novel_view
│     ├── sieve_novel_view
```
In every scene you need to change 'source_path' in cfg_args as follows:
```shell
source_path='<path to your dataset>'

# For example
source_path='/home/fansa/DS3DGS/data/NeRF-DS/as_novel_view'
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

