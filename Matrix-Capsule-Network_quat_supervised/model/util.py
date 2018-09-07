'''
Created on Sep 6, 2018

@author: jens
'''

from torchvision import datasets
import numpy as np
import torch

def print_mat(x):
    for i in range(x.size(1)):
        plt.matshow(x[0, i].data.cpu().numpy())

    plt.show()

def isnan(x):
    kaj = (x != x).sum()
    return (kaj != 0)

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
    mat = torch.cat((mat.view(q.shape[0],-1),torch.zeros(q.shape[0],3)),1)
    return mat

def matAffine_from_matMinRep(mat):
    m = mat[:,:9].view(mat.shape[0], 3, 3)
    mat_list = []
    for i in range(m.shape[0]):
        m = torch.cat([m[i,:,:2], torch.zeros(3,1), m[i,:,2:]], 1)
        z_axis = m[:,0].cross(m[:,1])
        m[:,2] = z_axis / z_axis.norm()
        m = torch.cat([m, torch.tensor([0.,0.,0.,1.]).view(1,4)], 0)
        mat_list.append(m)
    m = torch.stack(mat_list,dim=0)
    return m

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

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
