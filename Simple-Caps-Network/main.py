'''
Created on Aug 28, 2018

@author: jens
'''

import argparse
from model import simple
import time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simple_Caps-Network')
    parser.add_argument('--num-epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--recon-factor', type=float, default=0.005, metavar='N', help='use reconstruction loss or not')
    args = parser.parse_args()
    args.num_classes = 1
    args.batch_size = 1
    args.routing = 'EM_routing'
    args.loss = 'spread_loss'
    args.num_workers = 1
    args.no_labels = False
    args.disable_encoder = False
    args.disable_recon = False

    simple.run(args, verbose=True)
    print("done...")
