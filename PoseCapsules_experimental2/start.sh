python3 -m visdom.server &
python3 main.py --batch_size 25 --patience 100 --disable_recon --dataset 'rabbit400x400' --loss MSE --lr 5e-2
