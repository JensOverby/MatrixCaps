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
from scipy.linalg._flapack import dsfrk
random.seed(1991)
from PIL import Image
import math
from tqdm import tqdm
import os
import glob
from torchvision import transforms
import torch.nn as nn
import pyrr
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from collections import OrderedDict
#from axisAngle import get_y

#meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
#setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env='PoseCapsules')

def getSigmoidParams(criteria=9/10):
    dist = (torch.linspace(0,1, steps=1000) > criteria).float()
    mu_ref = dist.mean()
    var_ref = dist.std()**2
    kaj = torch.linspace(-1,1)
    kaj = kaj / kaj.std()
    A = 1
    B = 1
    while (True):
        c = 0
        mu = torch.sigmoid(-A + B*kaj).mean()
        if mu > mu_ref:
            A += 0.1
            c += 1
        var = torch.sigmoid(-A + B*kaj).std()**2
        if var < var_ref:
            B += 0.1
            c += 1
        if c == 0:
            break
    return A, B


class statBase():
    def __init__(self, args):
        self.args = args
        self.lossAvg = tnt.meter.AverageValueMeter()
        #self.lossSparseMu = tnt.meter.AverageValueMeter()
        #self.lossSparseVar = tnt.meter.AverageValueMeter()
        self.train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env='PoseCapsules')
        self.test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env='PoseCapsules')
        self.recon_sum = 0
        self.rout_id = 1
        if not self.args.disable_recon:
            self.reconLossAvg = tnt.meter.AverageValueMeter()
            self.ground_truth_logger_left = VisdomLogger('image', opts={'title': 'Ground Truth, left'}, env='PoseCapsules')
            self.reconstruction_logger_left = VisdomLogger('image', opts={'title': 'Reconstruction, left'}, env='PoseCapsules')
        if self.args.regularize:
            self.regularizeLossAvg = tnt.meter.AverageValueMeter()
            self.logsigAvg = tnt.meter.AverageValueMeter()
            self.costmeanAvg = tnt.meter.AverageValueMeter()
            self.costAvg = tnt.meter.AverageValueMeter()
            self.aAvg = tnt.meter.AverageValueMeter()
        
    def reset(self):
        self.lossAvg.reset()
        if not self.args.disable_recon:
            self.reconLossAvg.reset()
        if self.args.regularize:
            self.regularizeLossAvg.reset()
            self.logsigAvg.reset()
            self.costmeanAvg.reset()
            self.costAvg.reset()
            self.aAvg.reset()
        
    def log(self, pbar, output, labels, dict = OrderedDict(), stat=None):
        if not self.args.disable_loss:
            dict['loss'] = self.lossAvg.value()[0]
        if stat is not None:
            self.logsigAvg.add(stat[-self.rout_id*4 + 0])
            self.costmeanAvg.add(stat[-self.rout_id*4 + 1])
            self.costAvg.add(stat[-self.rout_id*4 + 2])
            self.aAvg.add(stat[-self.rout_id*4 + 3])
            stat.clear()
            dict['logsig'] = self.logsigAvg.value()[0]
            dict['costmean'] = self.costmeanAvg.value()[0]
            dict['cost'] = self.costAvg.value()[0]
            dict['a'] = self.aAvg.value()[0]
        #dict['mloss'] = self.lossSparseMu.value()[0]
        #dict['vloss'] = self.lossSparseVar.value()[0]
        if not self.args.disable_recon:
            #pbar.set_postfix(loss=self.lossAvg.value()[0], refresh=False)
            #else:
            dict['reconloss'] = self.reconLossAvg.value()[0]
            dict['rsum'] = self.recon_sum
            #pbar.set_postfix(loss=self.lossAvg.value()[0], rloss=self.reconLossAvg.value()[0], rsum=self.recon_sum, refresh=False)
        if self.args.regularize:
            dict['reguloss'] = self.regularizeLossAvg.value()[0]

        pbar.set_postfix(dict, refresh=False)

    def endTrainLog(self, epoch, groundtruth_image=None, recon_image=None):
        #self.train_loss = self.lossAvg.value()[0]
        if not self.args.disable_loss:
            self.train_loss_logger.log(epoch, self.lossAvg.value()[0], name='loss')
            with open("train.log", "a") as myfile:
                myfile.write(str(self.lossAvg.value()[0]) + '\n')
        if not self.args.disable_recon:
            if groundtruth_image is not None:
                self.ground_truth_logger_left.log(make_grid(groundtruth_image, nrow=int(self.args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
            if recon_image is not None:
                self.reconstruction_logger_left.log(make_grid(recon_image.data, nrow=int(self.args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
            #self.train_recon_loss = self.reconLossAvg.value()[0]
            self.train_loss_logger.log(epoch, self.reconLossAvg.value()[0], name='recon')
        #if self.args.regularize:
        #    self.train_regularize_loss = self.regularizeLossAvg.value()[0]
            
    def endTestLog(self, epoch):
        #loss = self.lossAvg.value()[0]
        if not self.args.disable_loss:
            self.test_loss_logger.log(epoch, self.lossAvg.value()[0], name='loss')
            with open("test.log", "a") as myfile:
                myfile.write(str(self.lossAvg.value()[0]) + '\n')
        if not self.args.disable_recon:
            self.test_loss_logger.log(epoch, self.reconLossAvg.value()[0], name='recon')

    def load_loss(self, history_count):
        if os.path.isfile('train.log'):
            with open("train.log", "r") as lossfile:
                loss_list = []
                for loss in lossfile:
                    loss_list.append(loss)
                while len(loss_list) > history_count:
                    loss_list.pop(0)
                epoch = -len(loss_list)
                for loss in loss_list:
                    self.train_loss_logger.log(epoch, float(loss), name='loss')
                    epoch += 1
        if os.path.isfile('test.log'):
            with open("test.log", "r") as lossfile:
                loss_list = []
                for loss in lossfile:
                    loss_list.append(loss)
                while len(loss_list) > history_count:
                    loss_list.pop(0)
                epoch = -len(loss_list)
                for loss in loss_list:
                    self.test_loss_logger.log(epoch, float(loss), name='loss')
                    epoch += 1

class statClassification(statBase):
    def __init__(self, args):
        super(statClassification, self).__init__(args)
        self.meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
        self.accuracy_logger = VisdomPlotLogger('line', opts={'title': 'accuracy'}, env='PoseCapsules')

    def reset(self):
        super(statClassification, self).reset()
        self.meter_accuracy.reset()

    def log(self, pbar, output, labels, stat=None):
        self.meter_accuracy.add(output.squeeze()[:,:,-1:].squeeze().data, labels.data)
        dict = OrderedDict()
        dict['acc'] = self.meter_accuracy.value()[0]
        super(statClassification, self).log(pbar, output, labels, dict, stat)

    def endTrainLog(self, epoch, groundtruth_image=None, recon_image=None):
        super(statClassification, self).endTrainLog(epoch, groundtruth_image, recon_image)
        self.accuracy_logger.log(epoch, self.meter_accuracy.value()[0], name='train')

    def endTestLog(self, epoch):
        super(statClassification, self).endTestLog(epoch)
        self.accuracy_logger.log(epoch, self.meter_accuracy.value()[0], name='test')
        print ("Test accuracy: ", self.meter_accuracy.value()[0])


class statJoints(statBase):
    def __init__(self, args, scale = [1.,1.,1.]):
        super(statJoints, self).__init__(args)
        self.jointErrAvg = tnt.meter.AverageValueMeter()
        self.scale = scale

    def reset(self):
        super(statJoints, self).reset()
        self.jointErrAvg.reset()

    def log(self, pbar, output, labels, stat=None):
        #err = (output[:,:4,0,0,:-1].data - labels[:,:4,:]) #.abs().mean().item()
        #err = err.view(err.shape[0], err.shape[1],-1, 3)
        shp = (labels.shape[0], labels.shape[1], -1, 3)
        
        labels_abs = labels.view(shp)
        output_abs = output.data[...,:-1].view(shp)
        
        """
        l_labels = [labels_abs[:,:,0,:]]
        l_output = [output_abs[:,:,0,:]]
        for i in range(4):
            l_labels.append( l_labels[i] + labels_abs[:,:,i+1,:] )
            l_output.append( l_output[i] + output_abs[:,:,i+1,:] )
        labels_abs = torch.stack(l_labels, dim=2)
        output_abs = torch.stack(l_output, dim=2)
        """

        #err = (output[...,:-1].data.view(shp)[:,:,1:,:] - labels.view(shp)[:,:,1:,:])
        err = output_abs - labels_abs
        sc = torch.from_numpy(self.scale).float().cuda()[None,None,None,:].expand(err.shape)   #torch.tensor([self.scale], device=output.device)[None, None, None, :].expand(err.shape)
        err = err * sc
        mean = err[:,:,1:,:].norm(dim=3).mean().item()
        mean1 = err[:,:,0,:].norm(dim=2).mean().item()
        mean = (20*mean + mean1)/21
        self.jointErrAvg.add(mean)
        dict = OrderedDict()
        dict['jointErr'] = self.jointErrAvg.value()[0]
        super(statJoints, self).log(pbar, output, labels, dict, stat)

class statTransmatrix(statBase):
    def __init__(self, args):
        super(statTransmatrix, self).__init__(args)
        self.angleErrAvg = tnt.meter.AverageValueMeter()
        self.xyzErrAvg = tnt.meter.AverageValueMeter()    

    def reset(self):
        super(statTransmatrix, self).reset()
        self.angleErrAvg.reset()
        self.xyzErrAvg.reset()

    def log(self, pbar, output, labels, stat=None):
        angleErr, xyzErr = get_error(output[:,0,0,0,:-1].data.cpu(), labels.data.cpu())
        self.angleErrAvg.add(angleErr)
        self.xyzErrAvg.add(xyzErr)
        if self.args.disable_recon:
            pbar.set_postfix(loss=self.lossAvg.value()[0], AngErr=self.angleErrAvg.value()[0], xyzErr=self.xyzErrAvg.value()[0], refresh=False)
        else:
            pbar.set_postfix(loss=self.lossAvg.value()[0], rloss=self.reconLossAvg.value()[0], AngErr=self.angleErrAvg.value()[0], xyzErr=self.xyzErrAvg.value()[0], recon_=self.recon_sum, refresh=False)


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
    mm = mat[:,:9].view(mat.shape[0], 3, 3)
    mm = mm.permute(0,2,1)
    mat_list = []
    for i in range(mm.shape[0]):
        m = torch.cat([mm[i,:,:2], torch.zeros(3,1), mm[i,:,2:]], 1)
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
    def __init__(self, root, transform=None, target_transform=None, data_rep='MSE'):
        super(MyImageFolder, self).__init__(root, transform, target_transform)
        self.data_rep = 0 if data_rep == 'MSE' else 1

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        data = path.split('/')
        data = data[-1].split('_')
        data[-1] = data[-1].split('.p')[0]
        data = [float(i) for i in data]

        if len(data) == 8:
            data = matMinRep_from_qvec(torch.tensor(data).unsqueeze(0)).squeeze()
            
        if self.data_rep==0:
            labels = torch.tensor(data)
        else:
            R = np.array(data[:6]).reshape(2,3)
            R = np.stack([R[0], R[1], np.cross(R[0],R[1])], axis=0)
            #axis_angle = get_y(R)
            Q = pyrr.Quaternion.from_matrix(R)
            axis_angle_rep = np.concatenate([Q.axis*Q.angle, np.array(data[6:9])], axis=0)
            labels = torch.from_numpy(axis_angle_rep).float()
        
        #labels = matMinRep_from_qvec(labels.unsqueeze(0)).squeeze()
        
        return sample, labels

#labels[:,None,3:7].shape
#labels[(labels[:,6] < 0).nonzero(),3:7].shape

class myTest(data.Dataset):
    def __init__(self, width=28, sz=1000, img_type='one_point', factor=0.3, rnd = True, transform=None, target_transform=None, max_z=-100000, min_z=100000):
        super(myTest, self).__init__()
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

                    img = img.unsqueeze(0)
                            
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
                    img = torch.zeros(3, width,width)

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
                        xyz = pyrr.vector3.create(math.cos(R)*5 + 2.5, math.sin(R)*5 + 2.5, 0.0)
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
                    
                    img[0, int(AA[0]), int(AA[1])] = red
                    img[1, int(BB[0]), int(BB[1])] = green
                    img[2, int(CC[0]), int(CC[1])] = blue

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
                    target = torch.tensor([mat[0,0], mat[0,1],mat[0,2],mat[1,0],mat[1,1],mat[1,2], xyz[0], xyz[1], xyz[2], 1.0])
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


"""
Loading weights of previously saved states and optimizer state
"""
def load_pretrained(model, optimizer, model_number, forced_lr, is_cuda, path="./weights/"):
    """
    Create "weights" folder for storing models
    """
    #weight_folder = 'weights'
    #if not os.path.isdir(weight_folder):
    #    os.mkdir(weight_folder)

    if model_number == -1: # latest
        model_name = max(glob.iglob("{}model*.pth".format(path)),key=os.path.getctime)
    else:
        model_name = "{}model_{}.pth".format(path, model_number)
        
    optim_name = "{}optim.pth".format(path)
    
    pretrained_dict = torch.load(model_name)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    #model.load_state_dict( torch.load(model_name) )
    
    if optimizer is not None and os.path.isfile(optim_name):
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

# function to get angle error between gt and predicted viewpoints
def get_error(yhat, ygt):
    if ygt.size(1) == 4:
        error = yhat - ygt
        return error[:,2:].median(), error[:,:2].median()
    else:
        xyz_idx = 6 if ygt.size(1)==10 else 3
        N = ygt.shape[0]
        az_error = np.zeros(N)
        for i in range(N):
            # read the 3-dim axis-angle vectors
            if ygt.size(1)==10:
                # MSE
                R1 = np.stack([ygt[i,:3], ygt[i,3:6], np.cross(ygt[i,:3],ygt[i,3:6])])
                R2 = np.stack([yhat[i,:3], yhat[i,3:6], np.cross(yhat[i,:3],yhat[i,3:6])])
            else:
                # GEO
                v1 = ygt[i,:3]
                v2 = yhat[i,:3]
                # get correponding rotation matrices
                R1 = pyrr.Matrix33.from_quaternion(pyrr.Quaternion.from_axis(v1))
                R2 = pyrr.Matrix33.from_quaternion(pyrr.Quaternion.from_axis(v2))
            #R1 = get_R(v1)
            #R2 = get_R(v2)
            # compute \|log(R_1^T R_2)\|_F/\sqrt(2) using Rodrigues' formula
            R = np.dot(R1.T, R2)
            tR = np.trace(R)
            theta = np.arccos(np.clip(0.5*(tR-1), -1.0, 1.0))   # clipping to avoid numerical issues
            atheta = np.abs(theta)
            # print('i:{0}, tR:{1}, theta:{2}'.format(i, tR, theta, atheta))
            az_error[i] = np.rad2deg(atheta)
        medErr = np.median(az_error)
        xyzErr = np.linalg.norm(yhat[:,xyz_idx:] - ygt[:,xyz_idx:], ord=1, axis=1)
        xyzErr = np.median(xyzErr)
        return medErr, xyzErr

def hookFunc(module, gradInput, gradOutput):
    #print(len(gradInput))
    for v in gradInput:
        print('Number of in NANs = ', (v != v).sum())
        #print (v)
    for v in gradOutput:
        print('Number of out NANs = ', (v != v).sum())
