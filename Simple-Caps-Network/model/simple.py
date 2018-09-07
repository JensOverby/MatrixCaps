'''
Created on Aug 28, 2018

@author: jens
'''

import matplotlib

import sys
sys.path.append('../Matrix-Capsule-Network/model')
import capsules as mcaps

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import glob
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib
from torchvision import utils
import pyrr

from model.data import images, labels

def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise

fig = None

def doDraw(img, i):
    image = img.squeeze() #.unsqueeze(1)
    if len(image.shape) > 2:
        image = image.unsqueeze(1)
        
    grid_picture = utils.make_grid(image.data, nrow=int(image.shape[0] ** 0.5), normalize=True,range=(0, 1)).cpu().numpy()
    axA = fig.add_subplot(1,6,i)
    if len(grid_picture.shape) > 2:
        axA.imshow(grid_picture.transpose(1,2,0))
    else:
        axA.imshow(grid_picture)
    return i+1

class CapsNet(nn.Module):
    def __init__(self, args, A=1, B=2, C=2, D=2, E=1, r=3, h=2):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=A, out_channels=B,kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=B, eps=0.001, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2,kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=2, eps=0.001, momentum=0.1, affine=True)
        self.primary_caps = mcaps.PrimaryCaps(B, C, h=h)
        self.convcaps1 = mcaps.ConvCaps(args, C, D, kernel=2, stride=1, h=h, iteration=r, coordinate_add=False, transform_share=False)
        self.classcaps = mcaps.ConvCaps(args, D, E, kernel=0, stride=1, h=h, iteration=r, coordinate_add=True, transform_share=True)

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
            i = doDraw(x[0][0], i)

        x = self.convcaps1(x, 1)
        if draw:
            y = x[0][0,0,0,0,:].view(2,2)
            i = doDraw(y, i)
            
        p,_ = self.classcaps(x, 1)
        if draw:
            y = p[0,0,0,0,:].view(2,2)
            i = doDraw(y, i)
            matplotlib.pyplot.show()

        p = p.squeeze()

        # Temporary when batch size = 1
        if len(p.shape) == 1:
            p = p.unsqueeze(0)
            
        #p = torch.cat([x[0], x[1].unsqueeze(-1)], dim=-1)
        #p = p.view(-1)

        if len(p.shape) == 1:
            p = p.unsqueeze(0)

        reconstructions = self.decoder(p)

        return p, reconstructions
    

def run(args, verbose=False):
    global fig
    
    model = CapsNet(args)
    model.cuda()
    #self = model.convcaps1
    
    capsule_loss = mcaps.CapsuleLoss(args)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True) #, eps=1e-3)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True)
    
    draw = False

    imgs = torch.from_numpy(images).float()
    #imgs = imgs.unsqueeze(0).unsqueeze(0)
    imgs = imgs.unsqueeze(1)
    #gaussian(imgs, 0.2, 0.1)
    imgs = Variable(imgs).cuda()
    
    l_ref = torch.from_numpy(labels).float()
    #l_ref = l_ref.view(l_ref.shape[0], -1).cuda()
    
    average_loss = 0
    for epoch in range(args.num_epochs):
        loss = 0
    
        if verbose and epoch==(args.num_epochs-1):
            draw=True
        
        #for i in range(images.shape[0]):
        optimizer.zero_grad()
    
        if draw:
            fig = plt.figure(figsize = (16,8))

        idx = int(epoch%(imgs.shape[0]/args.batch_size))
        begin = idx*args.batch_size
        batch_imgs = imgs[begin:begin+args.batch_size,...]
        batch_l_ref = l_ref[begin:begin+args.batch_size,...]
        rand_perm = torch.randperm(batch_imgs.shape[0])
        batch_imgs_perm = batch_imgs[rand_perm]
        batch_l_ref_perm = batch_l_ref[rand_perm]


        l_out, recon = model(batch_imgs_perm, draw)
        recon = recon.view_as(batch_imgs_perm)
    
        #l_out = out_labels #.view(-1)
        
    
        # Convert to quaternion
        vec4_list = []
        matrix = batch_l_ref_perm #torch.from_numpy(labels).float()
        for i in range(matrix.shape[0]):
            mat = torch.cat([matrix[i,:,:2], torch.zeros(3,1), matrix[i,:,2:]], 1)
            mat = torch.cat([mat[:2,:], torch.tensor([0.,0.,1.,0.]).view(1,4), mat[2:,:]], 0)
            quat = pyrr.Quaternion.from_matrix(mat.numpy())
            vec4 = np.array([quat[2],quat[3],mat[0,3],mat[1,3]])
            vec4_list.append( torch.from_numpy(vec4).view(-1) )
        batch_l_ref_perm = torch.stack(vec4_list,dim=0).cuda()

        
        loss = capsule_loss(batch_imgs_perm, l_out, batch_l_ref_perm, recon=recon)
    
        loss.backward()
        optimizer.step()
        average_loss += loss.data.item()
    
        if (epoch % 1000) == 999:
            print(epoch, "average = ", average_loss/1000)
            #scheduler.step(average_loss)
            average_loss = 0
        #print(epoch, loss.data.item())
            
        #if epoch == 2300:
        #    scheduler._reduce_lr(epoch)
    
    plt.show()
    print('loss =',loss.item())
