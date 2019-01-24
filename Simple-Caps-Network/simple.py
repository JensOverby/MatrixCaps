'''
Created on Aug 28, 2018

@author: jens
'''

import sys
sys.path.append('../MatrixCapsuleNetwork_supervised')
import model.capsules as mcaps
import model.util as util

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import utils
from data import testim, labels, features

def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise

fig = None

def doDraw(image, i):
    #image = img.squeeze() #.unsqueeze(1)
    if len(image.shape) > 2:
        image = image.unsqueeze(1)
        
    grid_picture = utils.make_grid(image.data, nrow=int(image.shape[0] ** 0.5), normalize=True,range=(0, 1)).cpu().numpy()
    axA = fig.add_subplot(1,6,i)
    if len(grid_picture.shape) > 2:
        axA.imshow(grid_picture.transpose(1,2,0))
    else:
        axA.imshow(grid_picture.reshape(-1,1))
    return i+1

class CapsNet(nn.Module):
    #def __init__(self, args, A=1, B=2, C=2, D=2, E=1, r=3, h=2):
    def __init__(self, args, A=1, B=32, C=4, D=2, E=1, r=3, h=2):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=A, out_channels=B,kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=B, eps=0.001, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(in_channels=B, out_channels=B,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=B, eps=0.001, momentum=0.1, affine=True)
        self.primary_caps = mcaps.PrimaryCaps(B, C, h=h)
        self.convcaps1 = mcaps.ConvCaps(C, D, kernel=3, stride=1, h=h, iteration=r, coordinate_add=False, transform_share=False)
        self.convcaps2 = mcaps.ConvCaps(D, E, kernel=0, stride=1, h=h, iteration=r, coordinate_add=True, transform_share=True)
        #self.classcaps = mcaps.ConvCaps(args, D, E, kernel=0, stride=1, h=h, iteration=r, coordinate_add=True, transform_share=True)

        """
        lin1 = nn.Linear(h*h*E, 32)
        lin2 = nn.Linear(32, 64)
        lin3 = nn.Linear(64, 64)
        
        self.decoder = nn.Sequential(
            lin1,
            nn.ReLU(inplace=True),
            lin2,
            nn.ReLU(inplace=True),
            lin3,
            nn.Sigmoid()
        )
        """

    def forward(self, x, draw=False):
        if draw:
            i = 1
            i = doDraw(x[0], i)

        x = F.relu(self.bn1(self.conv1(x)))
        if draw:
            i = doDraw(x[0], i)

        x = F.relu(self.bn2(self.conv2(x)))
        if draw:
            i = doDraw(x[0], i)

        x = self.primary_caps(x)
        if draw:
            i = doDraw(x[1][0], i)

        x = self.convcaps1(x, 1)
        if draw:
            i = doDraw(x[1][0], i)

        p,a = self.convcaps2(x, 1)
        if draw:
            i = doDraw(a[0], i)
            
        #p,_ = self.classcaps(x, 1)
        #if draw:
        #    y = p[0,0,0,0,:].view(2,2)
        #    i = doDraw(y, i)
        #    matplotlib.pyplot.show()

        p = p.squeeze()

        # Temporary when batch size = 1
        if len(p.shape) == 1:
            p = p.unsqueeze(0)
            
        #p = torch.cat([x[0], x[1].unsqueeze(-1)], dim=-1)
        #p = p.view(-1)

        #reconstructions = self.decoder(p)

        return p, None
    

def run(args, verbose=False):
    global fig
    
    model = CapsNet(args)
    if not args.disable_cuda:
        model.cuda()
    #self = model.convcaps1
    
    capsule_loss = mcaps.CapsuleLoss(args)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True) #, eps=1e-3)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True)
    
    draw = False

    #model.load_state_dict( torch.load("model.pth") )
    #torch.save(model.state_dict(), "model.pth")
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = 1e-3
    
    d = 8
    
    average_loss = 0
    mat_batch = None
    for epoch in range(args.num_epochs+1):
        loss = 0
    
        if verbose and epoch==(args.num_epochs):
            draw=True
            fig = plt.figure(figsize = (16,8))
            if not args.disable_cuda:
                mat_batch = Variable(torch.from_numpy(testim).float().unsqueeze(0)).cuda()
            else:
                mat_batch = Variable(torch.from_numpy(testim).float().unsqueeze(0))
            lab_batch = None

        else:
            if draw:
                fig = plt.figure(figsize = (16,8))
                
            idx = int(epoch%(features.shape[0]/args.batch_size))
            begin = idx*args.batch_size
            feature = features[begin:begin+args.batch_size,...]
            label = labels[begin:begin+args.batch_size,...]
            
            #begin = int(epoch%features.shape[0])
            #feature = features[begin:begin+1,...]
            #label = labels[begin:begin+1,...]

            mat_batch = []
            lab_batch = []
            for i in range(6):
                for j in range(6):
                    for k in range(args.batch_size):
                        #mat = np.array(np.zeros(d*d)).reshape(d,d)
                        mat = torch.zeros(d,d)
                        mat = util.gaussian(mat, 0.5, 0.5)
                        mat = (mat > 0.8).float()
                        mat[d-3-i:d-i, j:j+3] = torch.from_numpy(feature[k])

                        lab = np.array([[label[k,0,0], label[k,0,2]+j],
                                        [label[k,1,0], label[k,1,2]+i]])
        
                        if not args.disable_cuda:
                            #mat = Variable(torch.from_numpy(mat).float().view(1,d,d)).cuda()
                            mat = Variable(mat.view(1,d,d)).cuda()
                            lab = torch.from_numpy(lab).float().view(-1).cuda()
                        else:
                            mat = Variable(torch.from_numpy(mat).float().view(1,d,d))
                            lab = torch.from_numpy(lab).float().view(-1)
                        mat_batch.append(mat)
                        lab_batch.append(lab)
            mat_batch = torch.stack(mat_batch, dim=0)
            lab_batch = torch.stack(lab_batch, dim=0)
    
            rand_perm = torch.randperm(mat_batch.shape[0])
            mat_batch = mat_batch[rand_perm]
            lab_batch = lab_batch[rand_perm]            

            #mat_batch = util.gaussian(mat_batch, 0., 0.5)
            #mat_batch = torch.clamp(mat_batch, 0., 1.)


        optimizer.zero_grad()

        l_out, recon = model(mat_batch, draw)
        #recon = recon.view_as(mat_batch)
    
        if draw:
            print("out:", l_out)
            break
        
        loss = capsule_loss(mat_batch, l_out, lab_batch, recon=recon)
    
        loss.backward()
        optimizer.step()
        average_loss += loss.data.item()

    
        if (epoch % 100) == 99:
            print(epoch, "average = ", average_loss/(100*mat_batch.shape[0]))
            #scheduler.step(average_loss)
            average_loss = 0
        #print(epoch, loss.data.item())
            
        #if epoch == 2300:
        #    scheduler._reduce_lr(epoch)
    
    plt.show()
    #print('loss =',loss.item())
