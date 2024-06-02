export CUDA_VISIBLE_DEVICES=0

# python train.py -s data/NeRF-DS/as_novel_view -m output/NeRF-DS/as_novel_view --eval 
# python train.py -s data/NeRF-DS/basin_novel_view -m output/NeRF-DS/basin_novel_view --eval 
# python train.py -s data/NeRF-DS/bell_novel_view -m output/NeRF-DS/bell_novel_view --eval 
# python train.py -s data/NeRF-DS/cup_novel_view -m output/NeRF-DS/cup_novel_view --eval 
# python train.py -s data/NeRF-DS/plate_novel_view -m output/NeRF-DS/plate_novel_view --eval 
# python train.py -s data/NeRF-DS/press_novel_view -m output/NeRF-DS/press_novel_view --eval 
# python train.py -s data/NeRF-DS/sieve_novel_view -m output/NeRF-DS/sieve_novel_view --eval 
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
echo "Done"