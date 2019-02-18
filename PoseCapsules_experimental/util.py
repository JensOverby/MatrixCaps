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
random.seed(1991)
from PIL import Image
import math
from tqdm import tqdm
import os
import glob
from torchvision import transforms
import torch.nn as nn
import pyrr

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
        labels = matMinRep_from_qvec(labels.unsqueeze(0)).squeeze()
        
        #labels = torch.cat([labels[:3],labels[7:]])
        
        #if labels[6] < 0:
        #    labels[3:7] *= -1
        
        return index, sample, labels

#labels[:,None,3:7].shape
#labels[(labels[:,6] < 0).nonzero(),3:7].shape

class myTest(data.Dataset):
    def __init__(self, width=28, sz=1000, img_type='one_point', factor=0.3, rnd = True, transform=None, target_transform=None, max_z=-100000, min_z=100000):
        self.transform = transform
        self.target_transform = target_transform

        max_distance = math.sqrt(width**2 + width**2) * factor
        self.max_z = max_z
        self.min_z = min_z

        self.train_data = []
        self.train_labels = []
        
        with tqdm(total=sz) as pbar:
            if img_type=='one_dot':
                for _ in range(sz):
                    img = torch.zeros(width,width)
                    x = int(random.random()*width)
                    y = int(random.random()*width)
                    img[x,y] = 1
                    target = torch.FloatTensor((x,y))
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()
            elif img_type=='simple_angle':
                a = 0
                for _ in range(sz):
                    img = torch.zeros(width,width)
                    if rnd:
                        a = random.random()*2*math.pi
                    else:
                        a += 0.063
                    step_x = math.cos(a)
                    step_y = math.sin(a)
                    x = y = width/2
                    while True:
                        if x >= width or y >= width or x < 0 or y < 0:
                            break
                        img[int(y),int(x)] = 1
                        x += step_x
                        y -= step_y
                            
                    target = torch.FloatTensor((step_x,step_y))
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()
            elif img_type=='one_point':
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
            elif img_type=='one_point_rot':
                for _ in range(sz):
                    img = torch.zeros(width,width,2)
                    x = random.random()*(width-2)+1
                    y = random.random()*(width-2)+1
                    r = random.random()*math.pi/2 + math.pi/4
                    img[int(x),int(y),0] = 1 if math.cos(r) > 0 else 0
                    img[int(x),int(y),1] = math.sin(r)
                    target = torch.FloatTensor((x,y,math.cos(r),math.sin(r)))
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()
            elif img_type=='two_capsules':
                for _ in range(sz):
                    thetaA = random.random()*2*math.pi
                    vectorA = torch.tensor([random.random()*10, random.random()*10, math.cos(thetaA), math.sin(thetaA)])
                    vectorB = torch.tensor([vectorA[0]+math.cos(thetaA-math.pi/2)*4, vectorA[1]+math.sin(thetaA-math.pi/2)*4, math.cos(thetaA+math.pi/2), math.sin(thetaA+math.pi/2)])
                    target = torch.tensor([vectorA[0]+(vectorB[0]-vectorA[0])/2, vectorA[1]+(vectorB[1]-vectorA[1])/2, math.cos(thetaA-math.pi/2), math.sin(thetaA-math.pi/2)])
                    
                    thetaA = random.random()*2*math.pi
                    noiseA = torch.tensor([random.random()*10, random.random()*10, math.cos(thetaA), math.sin(thetaA)])
                    thetaA = random.random()*2*math.pi
                    noiseB = torch.tensor([random.random()*10, random.random()*10, math.cos(thetaA), math.sin(thetaA)])
                    
                    img = torch.stack([vectorA, vectorB, noiseA, noiseB]) # batch_size, input_dim, input_atoms, dim_x, dim_y
                    self.train_data.append(img[...,None, None])
                    self.train_labels.append(torch.tensor(target))
                    pbar.update()

                    """                    
                    x = random.random()*8
                    point1 = random.random()*10, random.random()*10, math.cos(theta), math.sin(theta)
                    point2 = random.random()*10, random.random()*10, math.cos(theta), math.sin(theta)
                    theta = random.random()*2*math.pi
                    point3 = x, x+2, math.cos(theta), math.sin(theta)
                    theta = random.random()*2*math.pi
                    point4 = x, x+2, math.cos(theta), math.sin(theta)
                    theta = random.random()*2*math.pi
                    noise1 = random.random()*10, random.random()*10, math.cos(theta), math.sin(theta)
                    theta = random.random()*2*math.pi
                    noise2 = random.random()*10, random.random()*10, math.cos(theta), math.sin(theta)

                    img = torch.stack([torch.tensor(point1), torch.tensor(point2), torch.tensor(point3), torch.tensor(point4), torch.tensor(noise1), torch.tensor(noise2)]) # batch_size, input_dim, input_atoms, dim_x, dim_y
                    target = point3[0]/2, point3[1]/2, point1[2]*2, point1[3]*2
                    self.train_data.append(img[...,None, None])
                    self.train_labels.append(torch.tensor(target))
                    pbar.update()
                    """
            elif img_type=='three_dot':
                scale = width/10.0
                A = np.array([-1,-2])
                B = np.array([1,-2])
                C = np.array([0,2])
                R = 0
                
                for _ in range(sz):
                    img = torch.zeros(width,width)
                    #XY = np.array([5., 5.])
                    XY = np.array([random.random()*5 + 2.5, random.random()*5 + 2.5])
                    
                    if rnd:
                        R = random.random()*2*math.pi
                    else:
                        R += 0.063
                        
                    AA = ( [A[0]*math.cos(R)-A[1]*math.sin(R), A[0]*math.sin(R)+A[1]*math.cos(R)] + XY ) * scale
                    BB = ( [B[0]*math.cos(R)-B[1]*math.sin(R), B[0]*math.sin(R)+B[1]*math.cos(R)] + XY ) * scale
                    CC = ( [C[0]*math.cos(R)-C[1]*math.sin(R), C[0]*math.sin(R)+C[1]*math.cos(R)] + XY ) * scale
                    Xaxis = [math.cos(R), math.sin(R)]

                    img[int(AA[0]), int(AA[1])] = 1
                    img[int(BB[0]), int(BB[1])] = 0.7
                    img[int(CC[0]), int(CC[1])] = 0.4
                            
                    target = torch.tensor([XY[0], XY[1], Xaxis[0], Xaxis[1]])
                    target[:2] = (target[:2] - 2.5) / (5./2.) - 1 # scale to {-1,1}
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()
            elif img_type=='three_dot_3d':
                scale = width/10.0
                A = np.array([-1,-2,0])
                B = np.array([1,-2,0])
                C = np.array([0,2,0])
                R = 0.01222

                if self.max_z != -100000 or self.min_z != 100000:
                    find_max_min = False
                else:
                    find_max_min = True
                for _ in range(sz):
                    img = torch.zeros(width,width,3)

                    if rnd:
                        x_rot = random.random()*2.0*math.pi
                        y_rot = random.random()*2.0*math.pi
                        #y_rot = (random.random()*0.5-0.25)*math.pi
                        z_rot = random.random()*2.0*math.pi
                        xyz = pyrr.vector3.create(random.random()*5 + 2.5, random.random()*5 + 2.5, 5.0)
                    else:
                        x_rot = R
                        y_rot = R
                        z_rot = R
                        xyz = pyrr.vector3.create(math.cos(R)*5 + 2.5, math.sin(R)*5 + 2.5, 5.0)
                        R += 0.063

                    euler = pyrr.euler.create(roll=x_rot, pitch=y_rot, yaw=z_rot)
                    #rot = pyrr.Quaternion.from_eulers(euler)
                    mat = pyrr.matrix33.create_from_eulers(euler)
                    AA = (pyrr.matrix33.apply_to_vector(mat, A)+xyz) * scale
                    BB = (pyrr.matrix33.apply_to_vector(mat, B)+xyz) * scale
                    CC = (pyrr.matrix33.apply_to_vector(mat, C)+xyz) * scale

                    red = AA[2] + 1.0*scale
                    green = BB[2] + 1.0*scale
                    blue = CC[2] + 1.0*scale
                    
                    img[int(AA[0]), int(AA[1]),0] = red
                    img[int(BB[0]), int(BB[1]),1] = green
                    img[int(CC[0]), int(CC[1]),2] = blue

                    if find_max_min:
                        if max([red,green,blue]) > self.max_z:
                            self.max_z = max([red,green,blue])
                        if min([red,green,blue]) < self.min_z:
                            self.min_z = min([red,green,blue])
                    
                    #axis = rot.axis
                    #modulo = int(rot.angle/math.pi)
                    #angle = rot.angle%math.pi
                    #if modulo%2 == 1:
                    #    axis *= -1
                    target = torch.tensor([xyz[0], xyz[1], xyz[2], mat[0,0], mat[0,1],mat[0,2],mat[1,0],mat[1,1],mat[1,2]])
                    #target[:3] = (target[:3] - 2.5) / (5./2.) - 1 # scale to {-1,1}
                    self.train_data.append(img)
                    self.train_labels.append(target)
                    pbar.update()
                
                a = (0.1-1)/(self.min_z-self.max_z)
                b = 1-self.max_z*a
                for i,val in enumerate(self.train_data):
                    bb = (val > 0).float() * b
                    aa = (val > 0).float() * a
                    self.train_data[i] *= aa
                    self.train_data[i] += bb
                    
            elif img_type=='three_point':
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
        
        """
        img = Image.fromarray((img.numpy()*255).astype(np.uint8)) #, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        """
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
