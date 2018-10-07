python3 -m visdom.server &
python3 main.py --pretrained --lr 1e-3 --patience 10 --load-loss 50 --bright-contrast --recon-factor 1e-8
