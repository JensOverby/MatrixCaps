# -*- coding: utf-8 -*-

'''
The Capsules layer.
@author: Yuxian Meng
'''
# TODO: use less permute() and contiguous()

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import random
import os
import glob
import torch.nn as nn
from model.transform import PoseToQuatNet

import argparse
from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)

def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise

class MyDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.dataset = []
        lines = tuple(open(filename, 'r'))
        for line in lines:
            data = line.lstrip('labels:  [').rstrip(']\n').split('] , [')
            output = [float(i) for i in data[0].split(',')]
            input = [float(i) for i in data[1].split(',')]
            self.dataset.append([torch.tensor(input), torch.tensor(output)])
        self.index = -1
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        self.index += 1
        return self.dataset[self.index]

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=500000)
    parser.add_argument('--lr',help='learning rate',type=float,nargs='?',const=0,default=None,metavar='PERIOD')    
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load-loss',help='Load prev loss',type=int,nargs='?',const=1000,default=None,metavar='PERIOD')    
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--pretrained',help='load pretrained epoch',type=int,nargs='?',const=-1,default=None,metavar='PERIOD')    
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--num-workers', type=int, default=2, metavar='N',
                        help='num of workers to fetch data')
    args = parser.parse_args()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()


    model = PoseToQuatNet()

    loss_function = nn.MSELoss(size_average=False)

    train_dataset = MyDataset("dataset.txt")

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)

    weight_folder = 'weights'
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    if not args.lr:
        learning_rate = 1e-2
    else:
        learning_rate = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) #, eps=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2000, factor=0.5, verbose=True)

    if args.pretrained is not None:
        
        if args.pretrained == -1: # latest
            model_name = max(glob.iglob("./weights/model*.pth"),key=os.path.getctime)
            #optim_name = max(glob.iglob("./weights/optim*.pth"),key=os.path.getctime)
        else:
            model_name = "./weights/{}.pth".format(args.pretrained)
            
        optim_name = "./weights/optim.pth"
        
        model.load_state_dict( torch.load(model_name) )
        if os.path.isfile(optim_name):
            optimizer.load_state_dict( torch.load(optim_name) )
            if args.lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

        # Temporary PyTorch bugfix: https://github.com/pytorch/pytorch/issues/2830
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        m = 0.8
        lambda_ = 0.9

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    loss_logger_loss = 0
    loss_logger_count = 0
    epoch_offset = 0
    if args.load_loss:
        if os.path.isfile('loss.log'):
            with open("loss.log", "r") as lossfile:
                loss_list = []
                for loss in lossfile:
                    loss_list.append(loss)
                while len(loss_list) > args.load_loss:
                    loss_list.pop(0)
                for loss in loss_list:
                    train_loss_logger.log(epoch_offset*args.print_freq, float(loss))
                    epoch_offset += 1
                epoch_offset -= 1


    with torch.cuda.device(args.gpu):
        if use_cuda:
            print("activating cuda")
            model.cuda()

        for epoch in range(args.num_epochs):

            # Train

            average_loss = 0
            size = 0

            for data in train_loader:
                if use_cuda:
                    x = data[0].cuda()
                    y = data[1].cuda()
                else:
                    x = data[0]
                    y = data[1]

                optimizer.zero_grad()

                #x = gaussian(x, 0.0, 0.01)

                out_labels = model(x)
                
                y[(y[:,6] < 0).nonzero(),3:] *= -1

                y = gaussian(y, 0.0, 0.01)
                
                #kaj[(kaj[:,6] < 0).nonzero()[0,0],:] *= -1
                
                # Normalize
                #out_labels_normalized = torch.cat([ out_labels[:,:3],
                #                                     out_labels[:,3:] / torch.norm(out_labels[:,3:], 2, 1, keepdim=True) ],
                #                                     dim=-1)
                                                     
                
                
                #out_labels_normalized = out_labels[:,3:] / torch.norm(out_labels[:,3:], 2, 1, keepdim=True)

                loss = loss_function(out_labels, y)

                loss.backward()

                optimizer.step()
                
                average_loss += loss.item()
                size += 1

            average_loss /= size
            print("Epoch {} ".format(epoch), "loss {0:.2f}".format(average_loss))

            loss_logger_loss += average_loss
            loss_logger_count += 1
            if args.print_freq == loss_logger_count:
                train_loss_logger.log(epoch+epoch_offset*args.print_freq, loss_logger_loss/loss_logger_count)
                with open("loss.log", "a") as myfile:
                    myfile.write(str(loss_logger_loss/loss_logger_count)+'\n')
                loss_logger_loss = 0
                loss_logger_count = 0
                

            #print(average_loss, size)
            scheduler.step(average_loss)

            if (epoch % 1000) == 999:
                torch.save(model.state_dict(), "./weights/model.pth")
                torch.save(optimizer.state_dict(), "./weights/optim.pth")
