python3 -m visdom.server &
python3 main.py --batch_size 50 --patience 3 --dataset 'smallNORB' --disable_recon --lr 5e-3 --pretrained
