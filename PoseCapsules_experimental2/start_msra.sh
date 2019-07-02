python3 -m visdom.server &
python3 main.py --batch_size 19 --patience 5 --dataset 'msra' --lr 1e-3 --regularize --disable_recon --pretrained --load_loss 10
