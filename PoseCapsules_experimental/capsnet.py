'''
Created on Jan 14, 2019

@author: jens
'''
import torch
import torch.nn as nn
from collections import OrderedDict

#from capsule_layers_pytorch import Mask, Length

import sys
sys.path.append("../DynamicRouting")
import layers
import math

class MSELossWeighted(nn.Module):
    def __init__(self, weight):
        super(MSELossWeighted, self).__init__()
        self.weight = weight
        
    def forward(self, input, target):
        pct_var = (input-target)**2
        out = pct_var * self.weight.expand_as(target)
        loss = out.sum() 
        return loss        


class DenseLayer(nn.Module):
    def __init__(self, layer_function, channel_dimension):
        super(DenseLayer, self).__init__()
        self.layer_function = layer_function
        self.channel_dimension = channel_dimension
        
    def forward(self, x):
        #y = getattr(self, self.layer_function)(x)
        y = self.layer_function(x.clone())
        y_concat = torch.cat([x, y], self.channel_dimension)
        return y_concat

class StoreObject():
    def __init__(self, value=None):
        self.value = value


class StoreLayer(nn.Module):
    def __init__(self, container):
        super(StoreLayer, self).__init__()
        self.container = container
        #self.not_initialized = True
        
    def forward(self, x):
        self.container.value = x.clone()
        return x

class ConcatLayer(nn.Module):
    def __init__(self, container, channel_dimension):
        super(ConcatLayer, self).__init__()
        self.channel_dimension = channel_dimension
        self.container = container
        
    def forward(self, x):
        y = torch.cat([self.container.value, x], self.channel_dimension)
        self.container.value = y.clone()
        return y

class BranchLayer(nn.Module):
    def __init__(self, primary_function, secondary_function, container):
        super(BranchLayer, self).__init__()
        self.primary_function = primary_function
        self.secondary_function = secondary_function
        self.container = container
        
    def forward(self, x):
        self.container.value = self.secondary_function(x.clone())
        return self.primary_function(x)


class CapsNet(nn.Module):
    def __init__(self, output_atoms, img_shape, dataset, data_rep='MSE', normalize=0, lambda_=0.0):
        super(CapsNet, self).__init__()
        self.normalize = normalize
        
        layer_list = OrderedDict()

        if dataset=='two_capsules':
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=2, h=4, num_routing=3, voting={'type': 'standard'})
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=1, h=output_atoms, num_routing=3, voting={'type': 'standard'})
        elif img_shape[-1] == 28:
            kernel_size = 7
            layer_list['primary'] = layers.CapsuleLayer(output_dim=8, h=4, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0})
            layer_list['conv1'] = layers.CapsuleLayer(output_dim=8, h=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': 0})
            layer_list['conv2'] = layers.CapsuleLayer(output_dim=8, h=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 4, 'padding': 0})
        elif img_shape[-1] == 100:
            #100->50->38->6
            """
            layer_list['conv1'] = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=21, stride=2, padding=10, bias=True)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2'] = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=2, padding=5, bias=True)
            layer_list['bn2'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu2'] = nn.ReLU(inplace=True)
            """

            sz = layers.calc_out(100, kernel=7, stride=2, padding=3)
            sz = layers.calc_out(sz, kernel=7, stride=1)
            sz = layers.calc_out(sz, kernel=7, stride=2, padding=3)
            sz = layers.calc_out(sz, kernel=7, stride=1)
            
            #100->50->44->22->16
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=2, padding=0, bias=True)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2'] = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, bias=True)
            layer_list['bn2'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu2'] = nn.ReLU(inplace=True)
            layer_list['conv3'] = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3, bias=True)
            layer_list['bn3'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu3'] = nn.ReLU(inplace=True)
            layer_list['conv2prim'] = layers.ConvToPrim()
            layer_list['prim1'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=1, voting={'type': 'Conv2d', 'sort': 64, 'stride': 1, 'kernel_size': 7, 'padding': 0})
            layer_list['prim2caps'] = layers.PrimToCaps()
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['caps2'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, h=output_atoms, num_routing=3, voting={'type': 'standard'})

            #self.image_decoder = layers.make_decoder( layers.make_decoder_list([output_dim, 512, 1024, 2048, 4096, img_shape[-1]**2 * 3], 'sigmoid') )

            """
            kernel_size = 21
            layer_list['primary'] = layers.CapsuleLayer(output_dim=8, h=8, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 17, 'padding': 0})
            layer_list['conv1'] = layers.CapsuleLayer(output_dim=8, h=8, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 9, 'padding': 0})
            layer_list['conv2'] = layers.CapsuleLayer(output_dim=8, h=16, num_routing=3, voting={'type': 'standard'})
            layer_list['conv3'] = layers.CapsuleLayer(output_dim=8, h=16, num_routing=3, voting={'type': 'standard'})
            layer_list['conv4'] = layers.CapsuleLayer(output_dim=1, h=16, num_routing=3, voting={'type': 'standard'})
            """
        elif img_shape[-1] == 400:
            sz = layers.calc_out(100, kernel=7, stride=2, padding=3)
            sz = layers.calc_out(sz, kernel=7, stride=1)
            sz = layers.calc_out(sz, kernel=7, stride=2, padding=3)
            sz = layers.calc_out(sz, kernel=7, stride=1)
            
            """
            --lr 1.25e-3
            """
            #100->50->44->22->16
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=8, kernel_size=15, stride=1, padding=7, bias=True)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=8, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2prim'] = layers.ConvToPrim()
            
            layer_list['prim1'] = layers.CapsuleLayer(output_dim=2, h=8, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 15, 'padding': 7})
            layer_list['prim2'] = layers.CapsuleLayer(output_dim=4, h=8, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 9, 'padding': 4})
            layer_list['prim3'] = layers.CapsuleLayer(output_dim=4, h=8, num_routing=3, voting={'type': 'Conv2d', 'sort': 512, 'stride': 2, 'kernel_size': 9, 'padding': 4})
            layer_list['prim2caps'] = layers.PrimToCaps()
            
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=40, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['dense_caps2'] = DenseLayer(layers.CapsuleLayer(output_dim=32, h=12, num_routing=3, voting={'type': 'standard'}), 1)
            layer_list['dense_caps3'] = DenseLayer(layers.CapsuleLayer(output_dim=24, h=12, num_routing=3, voting={'type': 'standard'}), 1)
            decoder_input_atoms = 10
            layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, h=decoder_input_atoms, num_routing=3, voting={'type': 'standard'})
            """
            layer_list['prim3'] = layers.CapsuleLayer(output_dim=4, h=16, num_routing=3, voting={'type': 'Conv2d', 'sort': 256, 'stride': 2, 'kernel_size': 7, 'padding': 0})
            layer_list['prim2matrix'] = layers.PrimToMatrixPrim()
            layer_list['matprim'] = layers.CapsuleLayer(output_dim=4, h=4, num_routing=0, voting={'type': 'prim_matrix'})
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=64, h=4, num_routing=3, voting={'type': 'matrix', 'lambda': lambda_})
            layer_list['caps2'] = layers.CapsuleLayer(output_dim=64, h=4, num_routing=3, voting={'type': 'matrix', 'lambda': lambda_})
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=64, h=4, num_routing=3, voting={'type': 'matrix', 'lambda': lambda_})
            layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, h=4, num_routing=3, voting={'type': 'matrix', 'lambda': lambda_})
            decoder_input_atoms = 16
            layer_list['capsToOut'] = layers.MatrixToOut(decoder_input_atoms)
            """
        elif img_shape[-1] == 401:
            sz = layers.calc_out(100, kernel=7, stride=2, padding=3)
            sz = layers.calc_out(sz, kernel=7, stride=1)
            sz = layers.calc_out(sz, kernel=7, stride=2, padding=3)
            sz = layers.calc_out(sz, kernel=7, stride=1)
            
            """
            --lr 1.25e-3
            """
            #100->50->44->22->16
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=12, kernel_size=7, stride=1, padding=0, bias=True)
            #layer_list['bn1'] = nn.BatchNorm2d(num_features=12, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2prim'] = layers.ConvToPrim()
            layer_list['prim1'] = layers.CapsuleLayer(output_dim=2, h=12, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0})
            layer_list['prim2'] = layers.CapsuleLayer(output_dim=4, h=12, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0})
            layer_list['prim3'] = layers.CapsuleLayer(output_dim=8, h=12, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0})
            layer_list['prim4'] = layers.CapsuleLayer(output_dim=8, h=12, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0})
            #layer_list['prim3'] = layers.CapsuleLayer(output_dim=8, h=12, num_routing=3, voting={'type': 'Conv2d', 'sort': 128, 'stride': 2, 'kernel_size': 7, 'padding': 0})
            layer_list['prim2caps'] = layers.PrimToCaps()
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['caps2'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, h=output_atoms, num_routing=3, voting={'type': 'standard'})
        elif img_shape[-1] == 20:
            if dataset == 'simple_angle':
                #20->14->9->4 
                layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=1, bias=True)
                layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
                layer_list['relu1'] = nn.ReLU(inplace=True)
                layer_list['conv2prim'] = layers.ConvToPrim()
                layer_list['prim1'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 6, 'padding': 0})
                layer_list['prim2caps'] = layers.PrimToCaps()
                layer_list['caps2'] = layers.CapsuleLayer(output_dim=1, h=output_atoms, num_routing=3, voting={'type': 'standard'})
                """
                layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=1, padding=0, bias=True)
                layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
                layer_list['relu1'] = nn.ReLU(inplace=True)
                layer_list['conv2prim'] = layers.ConvToPrim()
                sz = layers.calc_out(20, kernel=7, stride=1, padding=0)

                layer_list['primary'] = layers.CapsuleLayer(output_dim=8, h=4, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': sz, 'padding': 0})
                sz = layers.calc_out(sz, kernel=sz, stride=1)

                layer_list['caps1'] = layers.CapsuleLayer(output_dim=8, h=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 1, 'padding': 0})
                layer_list['prim2caps'] = layers.PrimToCaps()
                layer_list['caps2'] = layers.CapsuleLayer(output_dim=8, h=4, num_routing=3, voting={'type': 'standard'})
                layer_list['caps3'] = layers.CapsuleLayer(output_dim=1, h=output_dim, num_routing=3, voting={'type': 'standard'})
                """
            else:
                #20->14->9->4 
                layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=1, bias=True)
                layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
                layer_list['relu1'] = nn.ReLU(inplace=True)
                layer_list['conv2'] = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, stride=1, bias=True)
                layer_list['bn2'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
                layer_list['relu2'] = nn.ReLU(inplace=True)
                layer_list['conv2prim'] = layers.ConvToPrim()
                layer_list['prim1'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 6, 'padding': 0})
                layer_list['prim2caps'] = layers.PrimToCaps()
                layer_list['caps1'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
                layer_list['caps2'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
                layer_list['caps3'] = layers.CapsuleLayer(output_dim=16, h=12, num_routing=3, voting={'type': 'standard'})
                layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, h=output_atoms, num_routing=3, voting={'type': 'standard'})
        elif img_shape[-1] == 50:
            sz = layers.calc_out(50, kernel=15, stride=1, padding=7)
            sz = layers.calc_out(sz, kernel=15, stride=2, padding=7)
            sz = layers.calc_out(sz, kernel=9, stride=1, padding=4)
            sz = layers.calc_out(sz, kernel=9, stride=1, padding=4)

            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=8, kernel_size=15, stride=1, padding=7, bias=True)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=8, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2prim'] = layers.ConvToPrim()
            layer_list['prim1'] = layers.CapsuleLayer(output_dim=2, h=8, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 15, 'padding': 7})
            layer_list['prim2'] = layers.CapsuleLayer(output_dim=4, h=8, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 9, 'padding': 4})
            layer_list['prim3'] = layers.CapsuleLayer(output_dim=4, h=8, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 9, 'padding': 4})
            layer_list['prim2caps'] = layers.PrimToCaps()
            
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=8, h=12, num_routing=3, voting={'type': 'standard'})
            layer_list['dense_caps2'] = DenseLayer(layers.CapsuleLayer(output_dim=6, h=12, num_routing=3, voting={'type': 'standard'}), 1)
            layer_list['dense_caps3'] = DenseLayer(layers.CapsuleLayer(output_dim=4, h=12, num_routing=3, voting={'type': 'standard'}), 1)
            decoder_input_atoms = 10
            layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, h=decoder_input_atoms, num_routing=3, voting={'type': 'standard'})

        self.capsules = nn.Sequential(layer_list)

        if data_rep=='MSE':
            self.target_decoder = nn.BatchNorm1d(num_features=decoder_input_atoms)
        else:
            self.target_decoder = layers.make_decoder( layers.make_decoder_list([decoder_input_atoms, 512, 512, output_atoms], 'tanh') )
            self.target_decoder.add_module('scale_pi', layers.ScaleLayer(math.pi))

    def forward(self, x, y, disable_recon=False):

        conv_cap = self.capsules(x)

        p = conv_cap.view(conv_cap.size(0),-1)
        p = self.target_decoder(p)
        #p = self.batchnorm1d(p)
        #p = p/p.sum(1, keepdim=True)
        
        if self.normalize > 0:
            lenx = p[:,:3].norm(dim=1, p=2, keepdim=True)
            leny = p[:,3:6].norm(dim=1, p=2, keepdim=True)
            #act1 = (lenx.squeeze() > 0.2)*(leny.squeeze() > 0.2)
            A, B = (0.2-1)/0.2, 1
            lenx = torch.where(lenx > 0.2, lenx, A*lenx+B)
            leny = torch.where(leny > 0.2, leny, A*leny+B)

            #zeros = torch.zeros(p.shape[0])
            #ones = torch.ones(p.shape[0])
            #act1 = torch.where((lenx > 0.2)*(leny > 0.2)==1, ones, zeros)

            pnorm = p.new_full(p.size(), 1) #torch.ones_like(p).cuda(self.device)
            pnorm[:,:3] = lenx
            pnorm[:,3:6] = leny
            p = p / pnorm

            if self.normalize > 1:
                len_b = p[:,3:6].norm(dim=1, p=2, keepdim=True)
                pnorm = p.new_full(p.size(), 1) #torch.ones_like(p).cuda(self.device)
                a = p.data[:,:3]
                b = p.data[:,3:6]
                c = torch.cross(a,b,dim=1)
                #act2 = (c.norm(dim=1) > 0.2) * act1
                #act2 = act2.view(-1,1).expand(-1,3)
                b1 = torch.cross(c,a,dim=1)
                b1 = b1 / b1.norm(dim=1,keepdim=True,p=2)
                b1 = b1 * len_b
                #a1 = a / a.norm(dim=1,keepdim=True,p=2)
                #b_norm = b / b.norm(dim=1,keepdim=True,p=2)
                #c = c / c.norm(dim=1,keepdim=True,p=2)
                #update = b1 - b_norm
                #b1 = b_norm + 0.5*update
                #b1 /= b1.norm(dim=1,keepdim=True,p=2)
                #a1 = torch.cross(b1,c)
    
                #pnorm[:,:3] = torch.where(act2, a1 / a, pnorm[:,:3])
                pnorm[:,3:6] = b1/b #torch.where(act2, b1 / b, pnorm[:,3:6])
                
                p = p * pnorm


        # Temporary when batch size = 1
        if len(p.shape) == 1:
            p = p.unsqueeze(0)
        p = p.view(p.shape[0],-1)

        #out = None #self.target_decoder(p)

        if not disable_recon:
            reconstructions = self.image_decoder(p)
        else:
            reconstructions = torch.zeros(1)
            
        return p, reconstructions
