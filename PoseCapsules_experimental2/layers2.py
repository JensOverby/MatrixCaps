'''
Created on Jan 14, 2019

@author: jens
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class MaskLayer(nn.Module):
    def __init__(self, num_classes):
        super(MaskLayer, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x = x.squeeze()
        _, y = x[:,:,-1:].max(dim=1)
        y = y.squeeze()
        y = Variable(torch.eye(self.num_classes, device=x.device)).index_select(dim=0, index=y) # convert to one hot
        return (x[:,:,:-1] * y[:, :, None]).view(x.size(0), -1)


class StoreLayer(nn.Module):
    def __init__(self, container, do_clone=True):
        super(StoreLayer, self).__init__()
        container.append(None)
        self.container = container
        self.do_clone = do_clone
        
    def forward(self, x):
        if type(x) is tuple:
            x = x[0]
        if self.do_clone:
            self.container[0] = x.clone()
        else:
            self.container[0] = x
        shp = x.shape
        if len(shp) == 4:
            """ If previous was Conv2d """
            self.container[0] = self.container[0].permute(0,2,3,1).unsqueeze(1)
        return x

class SplitStereoReturnLeftLayer(nn.Module):
    def __init__(self, right_container):
        super(SplitStereoReturnLeftLayer, self).__init__()
        right_container.append(None)
        self.right_container = right_container

    def forward(self, x):
        self.right_container[0] = x[:,:,:,int(x.shape[-1]/2):]
        return x[:,:,:,:int(x.shape[-1]/2)]

class RandomizeLayer(nn.Module):
    def __init__(self):
        super(RandomizeLayer, self).__init__()

    def forward(self, x):
        x_orig = None
        if type(x) is tuple:
            x_orig = x
            x = x_orig[0]
            #x = x[0]
        shp = x.shape
        x = x.view(shp[0],shp[1],-1,shp[-1])
        idx = torch.randperm(x.shape[2])
        x = x[:,:,idx,:].view(shp)
        
        if x_orig is None:
            return x
        return x, x_orig[1], x_orig[2], x_orig[3]

class ConcatLayer(nn.Module):
    def __init__(self, container, do_clone=True, keep_original=False):
        super(ConcatLayer, self).__init__()
        self.container = container
        self.do_clone = do_clone
        self.keep_original = keep_original
        
    def forward(self, x):
        if type(x) is tuple:
            x0 = x[0]
        else:
            x0 = x
        if type(self.container[0]) is tuple:
            y0 = self.container[0][0]
        else:
            y0 = self.container[0]

        if x0.shape[2:].numel() != y0.shape[2:].numel():
            if len(x0.shape) > 4:
                if (x0.shape[-2] == x0.shape[-1]) or (x0.shape[-1] == 1):
                    x0 = x0.permute(0, 1, 3, 4, 2).contiguous()                    # batch_size, output_dim, dim_x, dim_y, h
                x0 = x0.view(x0.size(0), x0.size(1), -1, x0.size(-1))             # batch_size, output_dim,dim_x*dim_y, h
            if len(y0.shape) > 4:
                if (y0.shape[-2] == y0.shape[-1]) or (y0.shape[-1] == 1):
                    y0 = y0.permute(0, 1, 3, 4, 2).contiguous()           # batch_size, output_dim, h, dim_x, dim_y, h
                y0 = y0.view(y0.size(0), -1, y0.size(-1), 1, 1)         # batch_size, output_dim*dim_x*dim_y, h, 1, 1

        y = torch.cat([y0, x0], 1)
        
        if not self.keep_original:
            if self.do_clone:
                self.container[0] = y.clone()
            else:
                self.container[0] = y

        #if (type(x) is tuple) and (type(self.container[0]) is tuple):
        #if self.include_routes:
        #    route = torch.cat([self.container[0][1], x[1]], 1)
        #    y = (y_store, route)
        #elif type(x) is tuple:
        #    y = (y_store, x[1])

        return y

class ActivatePathway(nn.Module):
    def __init__(self, container):
        super(ActivatePathway, self).__init__()
        self.container = container
        
    def forward(self, x):
        return self.container[0]

class BNLayer(nn.Module):
    def __init__(self):
        super(BNLayer, self).__init__()
        self.not_initialized = True

    def forward(self, x):
        shp = x.shape
        xx = x[:,:,:,:shp[3]-1,:,:].contiguous().view(shp[0]*shp[1], shp[2]*(shp[3]-1), shp[4], shp[5])
        yy = x[:,:,:,shp[3]-1:,:,:]
        if self.not_initialized:
            self.batchnorm = nn.BatchNorm2d(num_features=xx.shape[1])
            if x.is_cuda:
                self.batchnorm.cuda()
            self.not_initialized = False
        xx = self.batchnorm(xx).view(shp[0], shp[1], shp[2], shp[3]-1, shp[4], shp[5])
        xx = torch.tanh(xx)
        yy = torch.sigmoid(yy)
        x = torch.cat([xx,yy], dim=3)
        return x

class SigmoidLayer(nn.Module):
    def __init__(self, begin=-1):
        super(SigmoidLayer, self).__init__()
        self.begin = begin

    def forward(self, x): # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
        shp = x.shape
        xx = x[:,:,:,:self.begin,:,:]
        yy = x[:,:,:,self.begin:,:,:]
        yy = torch.sigmoid(yy)
        x = torch.cat([xx,yy], dim=3)
        return x

class Pose2VectorRepLayer(nn.Module):
    def __init__(self):
        super(Pose2VectorRepLayer, self).__init__()

    def forward(self, x):
        activation = torch.ones(x.shape[0], device=x.device)
        x = torch.cat([x,activation.unsqueeze(-1)], dim=1)
        return x.view(x.shape[0], 1, 1, 1, -1)

class UpsampleLayer(nn.Module):
    def __init__(self, new_h, pos_embed=False):
        super(UpsampleLayer, self).__init__()
        self.new_h = new_h
        self.pos_embed = pos_embed
        #self.not_initialized = True

    def forward(self, x):
        x_orig = None
        if type(x) is tuple:
            x_orig = x
            x = x_orig[0]
        shp = list(x.shape)
        if len(shp) == 6:
            h = shp[3]-1
            shp[3] = self.new_h - h
            x_new = torch.cat([x[:,:,:,:h,:,:], torch.zeros(shp), x[:,:,:,h:,:,:]], dim=3)
            return x_new
        if (x.shape[-2] == x.shape[-1]) or (x.shape[-1] == 1):
            h = shp[2]-1
            shp[2] = self.new_h - h
            x_new = torch.cat([x[:,:,:h,:,:], torch.zeros(shp, device=x.device), x[:,:,h:,:,:]], dim=2)
            return x_new
        h = shp[-1]
        #dim = int(math.sqrt(h))
        shp[-1] = self.new_h - h
        
        if self.pos_embed:
            if x_orig is not None:
                dim_x, dim_y = x_orig[2], x_orig[3]
                y_coord = (x_orig[1] / dim_y).unsqueeze(-1).float() / dim_y
                x_coord = (x_orig[1] % 50).unsqueeze(-1).float() / dim_x
            else:
                dim_x, dim_y = shp[2], shp[3]
                y_coord = torch.arange(dim_y, device=x.device).unsqueeze(1).repeat(1,dim_x).float() / dim_y
                y_coord = y_coord[None,None,:,:].repeat(shp[0],shp[1],1,1)
                x_coord = torch.arange(dim_x, device=x.device).unsqueeze(0).repeat(dim_y,1).float() / dim_x
                x_coord = x_coord[None,None,:,:].repeat(shp[0],shp[1],1,1)
            shp[-1] -=2
            ins = torch.cat([x_coord[...,None], y_coord[...,None], torch.zeros(shp, device=x.device)], dim=-1)
        else:
            ins = torch.zeros(shp, device=x.device)
        
        x_new = torch.cat([x[:,:,:,:,:h-1], ins, x[:,:,:,:,h-1:]], dim=-1)

        return x_new
