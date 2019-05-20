python3 -m visdom.server &
python3 main.py --batch_size 10 --patience 3 --dataset 'msra' --lr 2.5e-2 --pretrained
