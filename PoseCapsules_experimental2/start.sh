python3 -m visdom.server &
python3 main.py --batch_size 25 --patience 10 --dataset 'rabbit100x100' --lr 2.5e-2 --disable_recon
