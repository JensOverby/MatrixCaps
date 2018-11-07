'''
Created on Sep 6, 2018

@author: jens
'''

from torchvision import datasets
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random

def print_mat(x):
    for i in range(x.size(1)):
        plt.matshow(x[0, i].data.cpu().numpy())

    plt.show()

def matMinRep_from_qvec(q):

    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qx, qy, qz, qw = q[:,3], q[:,4], q[:,5], q[:,6]

    sqw = qw**2
    sqx = qx**2
    sqy = qy**2
    sqz = qz**2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    #m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    #m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    #m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs

    c0 = torch.stack((m00, m01, m02), dim=1)
    c1 = torch.stack((m10, m11, m12), dim=1)
    #c2 = torch.stack((m20, m21, m22, torch.zeros(q.shape[0])), dim=1)
    c3 = torch.stack((q[:,0], q[:,1], q[:,2]), dim=1) #torch.ones(q.shape[0])
    mat = torch.stack((c0, c1, c3), dim=2)
    mat = torch.cat((mat.view(q.shape[0],-1), q[:,7].unsqueeze(1), torch.zeros(q.shape[0],2)),1)
    return mat

def matAffine_from_matMinRep(mat):
    m = mat[:,:9].view(mat.shape[0], 3, 3)
    mat_list = []
    for i in range(m.shape[0]):
        m = torch.cat([m[i,:,:2], torch.zeros(3,1), m[i,:,2:]], 1)
        z_axis = m[:,0].cross(m[:,1])
        m[:,2] = z_axis / z_axis.norm()
        m = torch.cat([m, torch.tensor([0.,0.,0.,mat[i,9]]).view(1,4)], 0)
        mat_list.append(m)
    m = torch.stack(mat_list,dim=0)
    return m

def matrixString_from_matAffine(mat):
    string = ""
    for i in range(mat.shape[1]):
        m = mat[0,i,:].numpy()
        string += "%.3f\t%.3f\t%.3f\t%.3f" % (m[0],m[1],m[2],m[3])
        if i != (mat.shape[1]-1):
            string += "\n"
    return string

def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise

def applyBrightnessAndContrast(img, bright, cont, is_training=True):
    if is_training:
        converted = 0.5*(1 - cont) + bright + cont * img
        return np.clip(converted, 0., 1.)
    return img

def split_in_channels(imgs):
    # Split in 2 stereo images, 2 color each
    left = imgs[:,[0,2],:,:int(imgs.shape[-1]/2)]
    right = imgs[:,[0,2],:,int(imgs.shape[-1]/2):]
    imgs_stereo = np.stack([left[:,0,:,:],left[:,1,:,:],right[:,0,:,:],right[:,1,:,:]], axis=1)
    return imgs_stereo

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        data = path.split('/')
        data = data[-1].split('_')
        data[-1] = data[-1].split('.p')[0]
        data = [float(i) for i in data]
        
        labels = np.asarray(data)
        
        return sample, labels

class Conv2dGeneral(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, hh=16, transposed=False):
        super(Conv2dGeneral, self).__init__()
        self.K = kernel_size
        self.stride = stride
        self.transposed = transposed
        self.h = int(hh ** 0.5)
        self.hh = hh
        self.W = nn.Parameter(torch.randn(1, in_channels*kernel_size*kernel_size, out_channels, 1, 1, self.h, self.h))
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1, 1))

    def forward(self, input):
        """
        batch size, channels, x, y
        """
        w_in = input.shape[2]
        
        if self.transposed:
            stride = 1
            w_out = self.K + (w_in-1)*self.stride
            width = self.K + w_out - 1
            #spc = int(width/self.K - 1)
            spc = int((width-w_in)/w_in)
            #pad = width - self.K*(1+spc)
            pad = width - w_in*spc - w_in
            pad_right = int(pad/2)
            pad_left = int((pad+1)/2)
            spacing = torch.zeros(input.shape[0], input.shape[1], input.shape[3], spc, self.hh).cuda()
            padding_left = torch.zeros(input.shape[0], input.shape[1], input.shape[3], pad_left, self.hh).cuda()
            padding_right = torch.zeros(input.shape[0], input.shape[1], input.shape[3], pad_right, self.hh).cuda()
            input_expanded = torch.cat([torch.cat([input[:,:,:,i:i+1,:], spacing], dim=3) for i in range(input.shape[3])], dim=3)
            input_expanded = torch.cat([padding_left,input_expanded,padding_right], dim=3)
            spacing = torch.zeros(input.shape[0], input.shape[1], spc, width, self.hh).cuda()
            padding_left = torch.zeros(input.shape[0], input.shape[1], pad_left, width, self.hh).cuda()
            padding_right = torch.zeros(input.shape[0], input.shape[1], pad_right, width, self.hh).cuda()
            input_expanded = torch.cat([torch.cat([input_expanded[:,:,i:i+1,:,:], spacing], dim=2) for i in range(input.shape[2])], dim=2)
            input = torch.cat([padding_left,input_expanded,padding_right], dim=2)
        else:
            stride = self.stride
            w_out = int((w_in - self.K) / self.stride + 1)
            
        poses = torch.stack([input[:,:,stride * i:stride * i + self.K, stride * j:stride * j + self.K,:] for i in range(w_out) for j in range(w_out)], dim=-2)
        poses = poses.view(poses.shape[0], poses.shape[1]*self.K*self.K, 1, w_out, w_out, self.h, self.h)
        votes = self.W @ poses
        output = votes.sum(1) + self.bias
        return output.view(output.shape[:4]+torch.Size([self.hh]))

        """
        w_in = y.shape[2]
        k = self.backward_pass[0].weight.cpu()
        K = k.shape[2]
        w_out = k.shape[2] + (w_in-1)*self.st
        
        l = []
        for i in range(w_in):
            for j in range(w_in):
                m = torch.cat([torch.zeros(k.shape[0], k.shape[1], k.shape[2],j*self.st),k,torch.zeros(k.shape[0], k.shape[1], k.shape[2],(w_in-j-1)*self.st)], dim=3)
                m = torch.cat([torch.zeros(k.shape[0], k.shape[1], i*self.st,m.shape[3]),m,torch.zeros(k.shape[0], k.shape[1], (w_in-i-1)*self.st,m.shape[3])], dim=2)
                l.append(m.view(m.shape[0], m.shape[1], -1))
        C = torch.cat(l, dim=2).view(m.shape[0], m.shape[1], l[0].shape[2], -1).permute(0,1,3,2).contiguous()

        votes = C[:,None,:,:,:,None].matmul(y.view(y.shape[0],y.shape[1],1,-1,1,1))
        output = votes.sum(1).sum(2).view(y.shape[0], -1, w_out,w_out)
        """
