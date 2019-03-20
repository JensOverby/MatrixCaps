python3 -m visdom.server &
python3 main.py --batch_size 25 --patience 25 --disable_recon --dataset 'images' --loss MSE --lr 5e-2 --clamp 1
