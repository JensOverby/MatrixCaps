python3 -m visdom.server &
python3 main.py --batch_size 150 --patience 5 --dataset 'MNIST' --recon_factor 0.000005 --lr 1e-3 --pretrained
