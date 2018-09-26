python3 -m visdom.server &
python3 main.py --pretrained --lr 1e-3 --load-loss 100 --bright-contrast --recon-factor 1e-5
