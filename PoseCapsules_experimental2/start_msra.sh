python3 -m visdom.server &
python3 main.py --batch_size 10 --patience 8 --dataset 'msra' --lr 5e-3 --regularize --disable_recon --pretrained --load_loss 14
