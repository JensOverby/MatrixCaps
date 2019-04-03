python3 -m visdom.server &
python3 main.py --batch_size 25 --patience 150 --disable_recon --dataset 'rabbit50x50' --loss MSE --lr 5e-2
