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


class CapsNet(nn.Module):
    def __init__(self, output_dim, img_shape, dataset, normalize=0, device=torch.device('cuda')):
        super(CapsNet, self).__init__()
        self.normalize = normalize
        self.device = device
        
        layer_list = OrderedDict()

        if dataset=='two_capsules':
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=2, output_atoms=4, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=1, output_atoms=output_dim, num_routing=3, voting={'type': 'standard'}, device=device)
        elif img_shape[-1] == 28:
            kernel_size = 7
            layer_list['primary'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0}, device=device)
            layer_list['conv1'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': 0}, device=device)
            layer_list['conv2'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 4, 'padding': 0}, device=device)
        elif img_shape[-1] == 100 or img_shape[-1] == 400:
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
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=2, padding=3, bias=True)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2'] = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, bias=True)
            layer_list['bn2'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu2'] = nn.ReLU(inplace=True)
            layer_list['conv3'] = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3, bias=True)
            layer_list['bn3'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu3'] = nn.ReLU(inplace=True)
            layer_list['conv2prim'] = layers.ConvToPrim()
            layer_list['prim1'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=1, voting={'type': 'Conv2d', 'sort': 256, 'stride': 1, 'kernel_size': 7, 'padding': 0}, device=device)
            layer_list['prim2caps'] = layers.PrimToCaps()
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['caps2'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, output_atoms=output_dim, num_routing=3, voting={'type': 'standard'}, device=device)

            #self.image_decoder = layers.make_decoder( layers.make_decoder_list([output_dim, 512, 1024, 2048, 4096, img_shape[-1]**2 * 3], 'sigmoid') )

            """
            kernel_size = 21
            layer_list['primary'] = layers.CapsuleLayer(output_dim=8, output_atoms=8, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 17, 'padding': 0}, device=device)
            layer_list['conv1'] = layers.CapsuleLayer(output_dim=8, output_atoms=8, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 9, 'padding': 0}, device=device)
            layer_list['conv2'] = layers.CapsuleLayer(output_dim=8, output_atoms=16, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['conv3'] = layers.CapsuleLayer(output_dim=8, output_atoms=16, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['conv4'] = layers.CapsuleLayer(output_dim=1, output_atoms=16, num_routing=3, voting={'type': 'standard'}, device=device)
            """
        elif img_shape[-1] == 20:
            if dataset == 'simple_angle':
                #20->14->9->4 
                layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=1, bias=True)
                layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
                layer_list['relu1'] = nn.ReLU(inplace=True)
                layer_list['conv2prim'] = layers.ConvToPrim()
                layer_list['prim1'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 6, 'padding': 0}, device=device)
                layer_list['prim2caps'] = layers.PrimToCaps()
                layer_list['caps2'] = layers.CapsuleLayer(output_dim=1, output_atoms=output_dim, num_routing=3, voting={'type': 'standard'}, device=device)
                """
                layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=16, kernel_size=7, stride=1, padding=0, bias=True)
                layer_list['bn1'] = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)
                layer_list['relu1'] = nn.ReLU(inplace=True)
                layer_list['conv2prim'] = layers.ConvToPrim()
                sz = layers.calc_out(20, kernel=7, stride=1, padding=0)

                layer_list['primary'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': sz, 'padding': 0}, device=device)
                sz = layers.calc_out(sz, kernel=sz, stride=1)

                layer_list['caps1'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 1, 'padding': 0}, device=device)
                layer_list['prim2caps'] = layers.PrimToCaps()
                layer_list['caps2'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=3, voting={'type': 'standard'}, device=device)
                layer_list['caps3'] = layers.CapsuleLayer(output_dim=1, output_atoms=output_dim, num_routing=3, voting={'type': 'standard'}, device=device)
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
                layer_list['prim1'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 6, 'padding': 0}, device=device)
                layer_list['prim2caps'] = layers.PrimToCaps()
                layer_list['caps1'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=3, voting={'type': 'standard'}, device=device)
                layer_list['caps2'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=3, voting={'type': 'standard'}, device=device)
                layer_list['caps3'] = layers.CapsuleLayer(output_dim=16, output_atoms=12, num_routing=3, voting={'type': 'standard'}, device=device)
                layer_list['caps4'] = layers.CapsuleLayer(output_dim=1, output_atoms=output_dim, num_routing=3, voting={'type': 'standard'}, device=device)
        elif img_shape[-1] == 10:
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0], out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=8, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
            layer_list['conv2'] = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)
            layer_list['bn2'] = nn.BatchNorm2d(num_features=8, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu2'] = nn.ReLU(inplace=True)
            layer_list['conv2prim'] = layers.ConvToPrim()
            layer_list['prim1'] = layers.CapsuleLayer(output_dim=8, output_atoms=8, num_routing=1, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 3, 'padding': 1}, device=device)
            layer_list['prim2caps'] = layers.PrimToCaps()
            layer_list['caps1'] = layers.CapsuleLayer(output_dim=8, output_atoms=8, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['caps2'] = layers.CapsuleLayer(output_dim=8, output_atoms=8, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['caps3'] = layers.CapsuleLayer(output_dim=1, output_atoms=output_dim, num_routing=3, voting={'type': 'standard'}, device=device)

        self.capsules = nn.Sequential(layer_list)
        self.batchnorm1d = nn.BatchNorm1d(num_features=output_dim)


    def forward(self, x, y, disable_recon=False):

        conv_cap = self.capsules(x)

        p = conv_cap.squeeze()
        p = self.batchnorm1d(p)
        #p = p/p.sum(1, keepdim=True)
        
        if self.normalize > 0:
            lenx = p[:,:3].norm(dim=1, p=2, keepdim=True)
            leny = p[:,3:6].norm(dim=1, p=2, keepdim=True)
            act1 = (lenx.squeeze() > 0.2)*(leny.squeeze() > 0.2)
            A, B = (0.2-1)/0.2, 1
            lenx = torch.where(lenx > 0.2, lenx, A*lenx+B)
            leny = torch.where(leny > 0.2, leny, A*leny+B)

            #zeros = torch.zeros(p.shape[0])
            #ones = torch.ones(p.shape[0])
            #act1 = torch.where((lenx > 0.2)*(leny > 0.2)==1, ones, zeros)

            pnorm = torch.ones_like(p).cuda(self.device)
            pnorm[:,:3] = lenx
            pnorm[:,3:6] = leny
            p = p / pnorm

            if self.normalize > 1:
                pnorm = torch.ones_like(p).cuda(self.device)
                a = p.data[:,:3]
                b = p.data[:,3:6]
                c = torch.cross(a,b,dim=1)
                act2 = (c.norm(dim=1) > 0.2) * act1
                act2 = act2.view(-1,1).expand(-1,3)
                b1 = torch.cross(c,a,dim=1)
                b1 = b1 / b1.norm(dim=1,keepdim=True,p=2)
                a1 = a / a.norm(dim=1,keepdim=True,p=2)
                b_norm = b / b.norm(dim=1,keepdim=True,p=2)
                c = c / c.norm(dim=1,keepdim=True,p=2)
                update = b1 - b_norm
                b1 = b_norm + 0.5*update
                b1 /= b1.norm(dim=1,keepdim=True,p=2)
                a1 = torch.cross(b1,c)
    
                pnorm[:,:3] = torch.where(act2, a1 / a, pnorm[:,:3])
                pnorm[:,3:6] = torch.where(act2, b1 / b, pnorm[:,3:6])
                
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
