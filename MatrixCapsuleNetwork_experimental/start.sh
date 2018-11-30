python3 -m visdom.server &
python3 main.py --pretrained --lr 1e-3 --load-loss 10 --disable-dae --bright-contrast --r 5 --batch-size 2
