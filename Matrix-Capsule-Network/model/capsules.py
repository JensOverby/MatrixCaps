'''
Created on Jun 11, 2018

@author: jens
'''
# TODO: use less permute() and contiguous()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np
import random
import matplotlib.pyplot as plt

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)


def print_mat(x):
    for i in range(x.size(1)):
        plt.matshow(x[0, i].data.cpu().numpy())

    plt.show()

def isnan(x):
    kaj = (x != x).sum()
    return (kaj != 0)

def qmat(q):
    
    #tmp = input[:, 3:7]
    #q = []
    #for e in tmp:
    #    q.append(e.div(torch.norm(e, 2)))

    #q = torch.stack(q, dim=0)
    #q0 = q[:, 0] # obs: ret disse hvis ovenstående
    #q1 = q[:, 1]
    #q2 = q[:, 2]
    #q3 = q[:, 3]
    
    q0 = q[:, 3] # obs: ret disse hvis ovenstående
    q1 = q[:, 4]
    q2 = q[:, 5]
    q3 = q[:, 6]
    
    m00 = 1 - 2*(q2*q2) - 2*(q3*q3)
    m10 = 2*(q1*q2) + 2*(q3*q0)
    m20 = 2*(q1*q3) - 2*(q2*q0)
    c0 = torch.stack((m00, m10, m20), dim=1)
    
    m01 = 2*(q1*q2) - 2*(q3*q0)
    m11 = 1 - 2*(q1*q1) - 2*(q3*q3)
    m21 = 2*(q2*q3) + 2*(q1*q0)
    c1 = torch.stack((m01, m11, m21), dim=1)
    
    m02 = 2*(q1*q3) + 2*(q2*q0)
    m12 = 2*(q2*q3) - 2*(q1*q0)
    m22 = 1 - 2*(q1*q1) - 2*(q2*q2)
    c2 = torch.stack((m02, m12, m22), dim=1)

    c3 = q[:, :3]

    m = torch.stack((c0, c1, c2, c3), dim=2)
    
    return m


class PrimaryCaps(nn.Module):
    """
    Primary Capsule layer is nothing more than concatenate several convolutional
    layer together.
    Args:
        A:input channel
        B:number of types of capsules.

    """

    def __init__(self, A=32, B=32, h=4):
        super(PrimaryCaps, self).__init__()
        self.B = B
        self.capsules_pose = nn.ModuleList([nn.Conv2d(in_channels=A, out_channels=h * h,
                                                      kernel_size=1, stride=1)
                                            for _ in range(self.B)])
        self.capsules_activation = nn.ModuleList([nn.Conv2d(in_channels=A, out_channels=1,
                                                            kernel_size=1, stride=1) for _
                                                  in range(self.B)])

    def forward(self, x):  # b,14,14,32
        poses = [self.capsules_pose[i](x) for i in range(self.B)]  # (b,16,12,12) *32
        poses = torch.cat(poses, dim=1)  # b,16*32,12,12
        activations = [self.capsules_activation[i](x) for i in range(self.B)]  # (b,1,12,12)*32
        activations = F.sigmoid(torch.cat(activations, dim=1))  # b,32,12,12
        return poses, activations


class ConvCaps(nn.Module):
    """
    Convolutional Capsule Layer.
    Args:
        B:input number of types of capsules.
        C:output number of types of capsules.
        kernel: kernel of convolution. kernel=0 means the capsules in layer L+1's
        receptive field contain all capsules in layer L. Kernel=0 is used in the
        final ClassCaps layer.
        stride:stride of convolution
        iteration: number of EM iterations
        coordinate_add: whether to use Coordinate Addition
        transform_share: whether to share transformation matrix.

    """

    def __init__(self, args, B=32, C=32, kernel=3, stride=2, h=4, iteration=3,
                 coordinate_add=False, transform_share=False):
        super(ConvCaps, self).__init__()
        self.B = B
        self.C = C
        self.K = kernel  # kernel = 0 means full receptive field like class capsules
        self.Bkk = None
        self.Cww = None
        self.b = args.batch_size
        self.stride = stride
        self.coordinate_add = coordinate_add
        self.transform_share = transform_share
        self.beta_v = nn.Parameter(torch.randn(self.C).view(1,self.C,1,1))
        self.beta_a = nn.Parameter(torch.randn(self.C).view(1,self.C,1))
        
        if not transform_share:
            self.W = nn.Parameter(torch.randn(B, kernel, kernel, C,
                                              h, h))  # B,K,K,C,4,4
        else:
            self.W = nn.Parameter(torch.randn(B, C, h, h))  # B,C,4,4

        self.iteration = iteration

        self.h = h
        self.hh = self.h*self.h
        self.routing = args.routing
        self.eps = 1e-10

    def coordinate_addition(self, width_in, votes):
        add = [[i / width_in, j / width_in] for i in range(width_in) for j in range(width_in)]  # K,K,w,w
        if self.W.is_cuda:
            add = torch.Tensor(add).cuda()
        else:
            add = torch.Tensor(add)
        add = Variable(add).view(1, 1, self.K, self.K, 1, 1, 1, 2)
        add = add.expand(self.b, self.B, self.K, self.K, self.C, 1, 1, 2).contiguous()
        votes[:, :, :, :, :, :, :, :2, -1] = votes[:, :, :, :, :, :, :, :2, -1] + add
        return votes

    #def down_w(self, w):
    #    return range(w * self.stride, w * self.stride + self.K)

    def EM_routing(self, lambda_, a_, V):
        # routing coefficient
        if self.W.is_cuda:
            R = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False).cuda() / self.Cww
        else:
            R = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False) / self.Cww

        for i in range(self.iteration):
            # M-step
            R = (R * a_).unsqueeze(-1)
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R).unsqueeze(1)
            sigma_square = ((R * (V - mu) ** 2).sum(1) / sum_R).unsqueeze(1)

            cost = (self.beta_v + torch.log(sigma_square.sqrt().view(self.b,self.C,-1,self.hh)+self.eps)) * sum_R.view(self.b, self.C,-1,1)
            a = torch.sigmoid(lambda_ * (self.beta_a - cost.sum(-1)))
            a = a.view(self.b, self.Cww)

            # E-step
            if i != self.iteration - 1:
                #mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a.data
                normal = Normal(mu, sigma_square.sqrt())
                p = torch.exp(normal.log_prob(V+self.eps))   # https://stackoverflow.com/questions/40472499/issue-nan-with-adam-solver
                ap = a[:,None,:] * p.sum(-1)
                R = Variable(ap / (torch.sum(ap, -1, keepdim=True) + self.eps), requires_grad=False)

        return a, mu

    def angle_routing(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.zeros([self.b, self.Bkk, self.Cww]), requires_grad=False).cuda()

        for i in range(self.iteration):
            R = F.softmax(R, dim=1)
            R = (R * a_)[..., None]
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R)[:, None, :, :]

            if i != self.iteration - 1:
                u_v = mu.permute(0, 2, 1, 3) @ V.permute(0, 2, 3, 1)
                u_v = u_v.squeeze().permute(0, 2, 1) / V.norm(2, -1) / mu.norm(2, -1)
                R = R.squeeze() + u_v
            else:
                sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R
                const = (self.beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R
                a = torch.sigmoid(lambda_ * (self.beta_a.repeat(self.b, 1) - const.sum(2)))

        return a, mu

    #torch.save(poses, 'poses.pt')

    def forward(self, x, lambda_):
        poses, activations = x
        width_in = poses.size(2)
        w = int((width_in - self.K) / self.stride + 1) if self.K else 1  # 5
        self.Cww = w * w * self.C
        #self.b = poses.size(0)

        if self.transform_share:
            if self.K == 0:
                self.K = width_in  # class Capsules' kernel = width_in
            W = self.W.view(self.B, 1, 1, self.C, self.h, self.h).expand(self.B, self.K, self.K, self.C, self.h, self.h).contiguous()
        else:
            W = self.W  # B,K,K,C,4,4

        self.Bkk = self.K * self.K * self.B

        # used to store every capsule i's poses in each capsule c's receptive field
        #pose = poses.contiguous()  # b,16*32,12,12
        pose = poses.view(self.b, self.hh, self.B, width_in, width_in).permute(0, 2, 3, 4, 1).contiguous()  # b,B,12,12,16
        poses = torch.stack([pose[:, :, self.stride * i:self.stride * i + self.K,
                             self.stride * j:self.stride * j + self.K, :] for i in range(w) for j in range(w)],
                            dim=-1)  # b,B,K,K,w*w,16
        poses = poses.view(self.b, self.B, self.K, self.K, 1, w, w, self.h, self.h)  # b,B,K,K,1,w,w,4,4
        W_hat = W[None, :, :, :, :, None, None, :, :]  # 1,B,K,K,C,1,1,4,4
        votes = W_hat @ poses  # b,B,K,K,C,w,w,4,4

        if self.coordinate_add:
            votes = self.coordinate_addition(width_in, votes)
            activations_ = activations.view(self.b, -1)[..., None].repeat(1, 1, self.Cww)
        else:
            activations_ = torch.stack([activations[:, :, self.stride * i:self.stride * i + self.K,
                                 self.stride * j:self.stride * j + self.K] for i in range(w) for j in range(w)],
                                dim=-1)  # b,B,K,K,w*w
            activations_ = activations_.view(self.b, self.Bkk, 1, -1).repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)
            
            #activations_ = [activations[:, :, self.down_w(x), :][:, :, :, self.down_w(y)]
            #                for x in range(w) for y in range(w)]
            #activation = torch.stack(
            #    activations_, dim=4).view(self.b, self.Bkk, 1, -1) \
            #    .repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)

        votes = votes.view(self.b, self.Bkk, self.Cww, self.hh)
        activations, poses = getattr(self, self.routing)(lambda_, activations_, votes)
        
        if isnan(activations) or isnan(poses):
            print("nan")
        
        return poses.view(self.b, self.C, w, w, -1), activations.view(self.b, self.C, w, w)


class CapsNet(nn.Module):
    def __init__(self, args, A=32, AA=32, B=32, C=32, D=32, E=10, r=3, h=4):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=A,
                               kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=A, out_channels=AA,
                               kernel_size=5, stride=2, padding=1)
        self.primary_caps = PrimaryCaps(AA, B, h=h)
        self.convcaps1 = ConvCaps(args, B, C, kernel=3, stride=2, h=h, iteration=r,
                                  coordinate_add=False, transform_share=False)
        self.convcaps2 = ConvCaps(args, C, D, kernel=3, stride=1, h=h, iteration=r,
                                  coordinate_add=False, transform_share=False)
        self.classcaps = ConvCaps(args, D, E, kernel=0, stride=1, h=h, iteration=r,
                                  coordinate_add=True, transform_share=True)

        lin1 = nn.Linear(h*h * args.num_classes, 512)
        #lin1 = nn.Linear(3*4 * args.num_classes, 512)
        lin1.weight.data *= 50.0 # inialize weights strongest here!
        lin2 = nn.Linear(512, 4096)
        lin2.weight.data *= 0.1
        lin3 = nn.Linear(4096, 20000)
        lin3.weight.data *= 0.1

        
        self.decoder = nn.Sequential(
            #nn.Linear(h*h * args.num_classes, 1024),   # 16 - 1024 - 10240 - 10000
            lin1,
            nn.ReLU(inplace=True),
            #nn.Linear(1024, 10240),
            lin2,
            nn.ReLU(inplace=True),
            #nn.Linear(10240, 10000),
            lin3,
            nn.Sigmoid()
        )

        self.args = args

        """
        elem_counter = 0
        for elems in self.decoder.children():
            if (elem_counter % 2) == 0:
                elems.weight *= 5.0
            elem_counter += 1
        """

        
        self.num_classes = E
        
    def forward(self, x, lambda_, labels=None):  # b,1,28,28
        if not self.args.disable_encoder:
            x = F.relu(self.conv1(x))  # b,32,12,12
            x = F.max_pool2d(x,2, 2, 1)
            x = F.relu(self.conv2(x))  # b,32,12,12
            x = self.primary_caps(x)  # b,32*(4*4+1),12,12
            x = self.convcaps1(x, lambda_)  # b,32*(4*4+1),5,5
            x = self.convcaps2(x, lambda_)  # b,32*(4*4+1),3,3
            p, a = self.classcaps(x, lambda_)  # b,10*16+10
    
            p = p.squeeze()
            
            # Temporary when batch size = 1
            if len(p.shape) == 1:
                p = p.unsqueeze(0)
        else:
            p = labels

        # convert to one hot
        #y = Variable(torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)
        if not self.args.disable_recon:
            reconstructions = self.decoder(p)
            #if labels is None:
            #    reconstructions = self.decoder(p)
            #else:
            #    labels44 = qmat(p).view(p.shape[0],-1)
            #    reconstructions = self.decoder(labels44)
        else:
            reconstructions = 0

        return p, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self, args):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.loss = nn.MSELoss(size_average=False) #args.loss
        #self.use_recon = args.use_recon
        #self.no_labels = args.no_labels
        self.args = args

    @staticmethod
    def spread_loss(x, target, m):  # x:b,10 target:b
        loss = F.multi_margin_loss(x, target, p=2, margin=m)
        return loss

    @staticmethod
    def cross_entropy_loss(x, target, m):
        loss = F.cross_entropy(x, target)
        return loss

    @staticmethod
    def margin_loss(x, labels, m):
        left = F.relu(0.9 - x, inplace=True) ** 2
        right = F.relu(x - 0.1, inplace=True) ** 2

        labels = Variable(torch.eye(args.num_classes).cuda()).index_select(dim=0, index=labels)

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        return margin_loss * 1/x.size(0)

    def forward(self, images, output=None, labels=None, recon=None):
        #main_loss = getattr(self, self.loss)(output, labels, m)
        if self.args.disable_encoder or labels == None:
            main_loss = 0
        else:
            main_loss = self.loss(output, labels)

        if not self.args.disable_recon:
            recon_loss = self.reconstruction_loss(recon, images)
            main_loss += self.args.recon_factor * recon_loss

        return main_loss
