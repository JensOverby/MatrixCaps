'''
Created on Sep 6, 2018

@author: jens
'''

from torchvision import datasets
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
import random
from PIL import Image
import math
from tqdm import tqdm
import os
import glob
from torchvision import transforms
import torch.nn as nn

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
    mat = torch.stack((c0, c1, c3), dim=1)
    mat = torch.cat((mat.view(q.shape[0],-1), q[:,7].unsqueeze(1)),1)
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

#def applyBrightnessAndContrast(img, bright, cont, is_training=True):
#    if is_training:
#        converted = 0.5*(1 - cont) + bright + cont * img
#        return np.clip(converted, 0., 1.)
#    return img

def split_in_channels(imgs):
    # Split in 2 stereo images, 2 color each
    left = imgs[:,[0,2],:,:int(imgs.shape[-1]/2)]
    right = imgs[:,[0,2],:,int(imgs.shape[-1]/2):]
    imgs_stereo = np.stack([left[:,0,:,:],left[:,1,:,:],right[:,0,:,:],right[:,1,:,:]], axis=1)
    return imgs_stereo

class RandomBrightnessContrast(nn.Module):
    def __init__(self, brightness=0, contrast=0):
        super(CapsNet, self).__init__()
        self.brightness_contrast = transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast)
        self.toPil = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()

    def forward(self, img):
        img = self.toPil(img)
        img = self.brightness_contrast(img)
        return self.toTensor(img)

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
        
        labels = torch.tensor(data)
        
        return index, sample, labels

class myTest(data.Dataset):
    def __init__(self, width=28, sz=1000, img_type='one_point', factor=0.3, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        max_distance = math.sqrt(width**2 + width**2) * factor

        self.train_data = []
        self.train_labels = []
        
        with tqdm(total=sz) as pbar:
            if img_type=='one_point':
                for _ in range(sz):
                    img = torch.zeros(width,width)
                    x = int(random.random()*width)
                    y = int(random.random()*width)
                    for i in range(width):
                        for j in range(width):
                            intensency = 1. - math.sqrt((x-i)**2 + (y-j)**2) / max_distance
                            if intensency < 0.:
                                intensency = 0.
                            img[i,j] = intensency**2
                            
                    target = torch.FloatTensor((x,y))
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()
            else:
                scale = width/10.0
                A = np.array([-1,-2])
                B = np.array([1,-2])
                C = np.array([0,2])
                
                for _ in range(sz):
                    img = torch.zeros(width,width)
                    XY = np.array([random.random()*5 + 2.5, random.random()*5 + 2.5])
                    #x = int(random.random()*5) + 2.5
                    #y = int(random.random()*5) + 2.5
                    R = random.random()*2*math.pi
                    AA = ( [A[0]*math.cos(R)-A[1]*math.sin(R), A[0]*math.sin(R)+A[1]*math.cos(R)] + XY ) * scale
                    BB = ( [B[0]*math.cos(R)-B[1]*math.sin(R), B[0]*math.sin(R)+B[1]*math.cos(R)] + XY ) * scale
                    CC = ( [C[0]*math.cos(R)-C[1]*math.sin(R), C[0]*math.sin(R)+C[1]*math.cos(R)] + XY ) * scale
                    Xaxis = [math.cos(R), math.sin(R)]
                    for i in range(width):
                        for j in range(width):
                            dist = (AA[0]-i)**2 + (AA[1]-j)**2
                            dist_next = (BB[0]-i)**2 + (BB[1]-j)**2
                            if dist_next < dist:
                                dist=dist_next
                            dist_next = (CC[0]-i)**2 + (CC[1]-j)**2
                            if dist_next < dist:
                                dist=dist_next
                            
                            intensency = 1. - math.sqrt(dist) / max_distance
                            if intensency < 0.7:
                                intensency = 0.
                            img[i,j] = intensency**2
                            
                    target = torch.tensor([XY[0], XY[1], Xaxis[0], Xaxis[1]])
                    target[:2] = (target[2] - 2.5) / (5./2.) - 1 # scale to {-1,1}
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        img, target = self.train_data[index], self.train_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        img = Image.fromarray((img.numpy()*255).astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return index, img, target


def load_loss(train_loss_logger, history_count):
    epoch_offset = 0
    if os.path.isfile('loss.log'):
        with open("loss.log", "r") as lossfile:
            loss_list = []
            for loss in lossfile:
                loss_list.append(loss)
            while len(loss_list) > history_count:
                loss_list.pop(0)
            for loss in loss_list:
                train_loss_logger.log(epoch_offset, float(loss))
                epoch_offset += 1
    return epoch_offset

"""
Loading weights of previously saved states and optimizer state
"""
def load_pretrained(model, optimizer, model_number, forced_lr, is_cuda):
    """
    Create "weights" folder for storing models
    """
    weight_folder = 'weights'
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    if model_number == -1: # latest
        model_name = max(glob.iglob("./weights/model*.pth"),key=os.path.getctime)
    else:
        model_name = "./weights/model_{}.pth".format(model_number)
        
    optim_name = "./weights/optim.pth"
    
    model.load_state_dict( torch.load(model_name) )
    if os.path.isfile(optim_name):
        optimizer.load_state_dict( torch.load(optim_name) )
        if forced_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = forced_lr

        # Temporary PyTorch bugfix: https://github.com/pytorch/pytorch/issues/2830
        if is_cuda:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    return None
