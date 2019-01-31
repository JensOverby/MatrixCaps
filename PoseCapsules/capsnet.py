'''
Created on Jan 14, 2019

@author: jens
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#from capsule_layers_pytorch import Mask, Length

import sys
sys.path.append("../DynamicRouting")
import layers


class CapsNet(nn.Module):
    def __init__(self, constrained=False, image_width=28, device=torch.device('cuda')):
        super(CapsNet, self).__init__()
        self.image_width = image_width
        if image_width == 28:
            kernel_size = 7
        elif image_width == 100:
            kernel_size = 21
        
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, stride=1, bias=True)
        nn.init.normal_(self.conv1.weight.data, mean=0,std=5e-2)
        nn.init.constant_(self.conv1.bias.data, val=0.1)

        #self.batchnorm = nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.1, affine=True)

        layer_list = OrderedDict()

        if constrained:
            if image_width == 28:
                layer_list['primary'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 7, 'padding': 0}, device=device)
                layer_list['conv1'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': 0}, device=device)
                layer_list['conv2'] = layers.CapsuleLayer(output_dim=8, output_atoms=4, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 4, 'padding': 0}, device=device)
            elif image_width == 100:
                layer_list['primary'] = layers.CapsuleLayer(output_dim=16, output_atoms=10, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 17, 'padding': 0}, device=device)
                layer_list['conv1'] = layers.CapsuleLayer(output_dim=16, output_atoms=10, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 9, 'padding': 0}, device=device)
                layer_list['conv2'] = layers.CapsuleLayer(output_dim=16, output_atoms=10, num_routing=3, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': 0}, device=device)
                layer_list['conv3'] = layers.CapsuleLayer(output_dim=16, output_atoms=10, num_routing=3, voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 4, 'padding': 0}, device=device)
            
        else:
            layer_list['primary'] = layers.CapsuleLayer(output_dim=1, output_atoms=1, num_routing=1, voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': 0}, device=device)
            layer_list['reshape'] = layers.Reshape()
            layer_list['conv1'] = layers.CapsuleLayer(output_dim=1, output_atoms=2, num_routing=3, voting={'type': 'standard'}, device=device)
            layer_list['conv2'] = layers.CapsuleLayer(output_dim=1, output_atoms=2, num_routing=3, voting={'type': 'standard'}, device=device)
    
        self.capsules = nn.Sequential(layer_list)

        self.target_decoder = layers.make_decoder( layers.make_decoder_list([16*10, 512, 512, 10], 'tanh') )

        self.image_decoder = layers.make_decoder( layers.make_decoder_list([10, 1024, 4096, image_width*image_width*3], 'sigmoid') )
        

    def forward(self, x, disable_recon=False):
        # Layer 1: Just a conventional Conv2D layer
        conv1 = self.conv1(x)

        """
        global_info = torch.arange(0.,0.5,0.5/conv1.shape[3]).cuda()
        conv1[:,:1,:,:] = conv1[:,:1,:,:] + global_info
        conv1[:,1:2,:,:] = conv1[:,1:2,:,:] + global_info.unsqueeze(-1)
        """
        
        #conv1 = self.batchnorm(conv1)
        conv1 = F.relu(conv1)
        conv1 = conv1.unsqueeze(1)
    
        
        
    
        conv_cap = self.capsules(conv1)

        p = conv_cap.squeeze()
        
        # Temporary when batch size = 1
        if len(p.shape) == 1:
            p = p.unsqueeze(0)
        p = p.view(p.shape[0],-1)

        out = self.target_decoder(p)

        if not disable_recon:
            reconstructions = self.image_decoder(out)
        else:
            reconstructions = torch.zeros(1)
            
        return out, reconstructions

        
    """
    def __init__(self, device):
        super(CapsNet, self).__init__()
        # Layer 1: Just a conventional Conv2D layer
        d = 100
        padding, d = calc_same_padding(d, kernel=33, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=33, stride=2, padding=padding, bias=True)
        nn.init.normal_(self.conv1.weight.data, mean=0,std=5e-2)
        nn.init.constant_(self.conv1.bias.data, val=0.1)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        padding, d = calc_same_padding(d, kernel=17, stride=2)
        self.primary_caps = ConvCapsuleLayer(output_dim=32, input_atoms=256, output_atoms=16, num_routing=1, stride=2, kernel_size=17, padding=padding, device=device)
    
        # Layer 2: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=9, stride=2)
        self.conv_cap_2 = ConvCapsuleLayer(output_dim=32, input_atoms=16, output_atoms=16, num_routing=3, stride=2, kernel_size=9, padding=padding, device=device)
    
        # Layer 3: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=2)
        self.conv_cap_3 = ConvCapsuleLayer(output_dim=32, input_atoms=16, output_atoms=16, num_routing=3, stride=2, kernel_size=5, padding=padding, device=device)
    
        # Layer 4: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=2)
        self.conv_cap_4 = ConvCapsuleLayer(output_dim=32, input_atoms=16, output_atoms=16, num_routing=3, stride=2, kernel_size=5, padding=padding, device=device)
    
        # Layer 5: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=3, stride=2)
        self.conv_cap_5 = ConvCapsuleLayer(output_dim=32, input_atoms=16, output_atoms=16, num_routing=3, stride=2, kernel_size=3, padding=padding, device=device)

        # Layer 6: Convolutional Capsule
        self.conv_cap_6 = ConvCapsuleLayer(output_dim=1, input_atoms=16, output_atoms=16, num_routing=3, stride=1, kernel_size=2, padding=0, device=device)
    
    
        self.decoder = make_decoder([16, 1024, 4096, 10000]) 
        

    def forward(self, x, disable_recon=False):
        # Layer 1: Just a conventional Conv2D layer
        conv1 = F.relu(self.conv1(x))
        conv1 = conv1.unsqueeze(1)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = self.primary_caps(conv1)
    
        # Layer 2: Convolutional Capsule
        conv_cap_2 = self.conv_cap_2(primary_caps)
    
        # Layer 3: Convolutional Capsule
        conv_cap_3 = self.conv_cap_3(conv_cap_2)
        #conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same', routings=3, name='conv_cap_3_1')(conv_cap_2_2)
    
        # Layer 4: Convolutional Capsule
        conv_cap_4 = self.conv_cap_4(conv_cap_3)
        #conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same', routings=3, name='conv_cap_4_1')(conv_cap_3_2)

        conv_cap_5 = self.conv_cap_5(conv_cap_4)

        conv_cap_6 = self.conv_cap_6(conv_cap_5)

        p = conv_cap_6.squeeze()
        
        # Temporary when batch size = 1
        if len(p.shape) == 1:
            p = p.unsqueeze(0)

        if not disable_recon:
            reconstructions = self.decoder(p)
            #if labels is None:
            #    reconstructions = self.decoder(p)
            #else:
            #    labels44 = qmat(p).view(p.shape[0],-1)
            #    reconstructions = self.decoder(labels44)
        else:
            reconstructions = torch.zeros(1)

        return p, reconstructions
    """
"""    
nn.init.normal(self.recon_1.weight, mean=0,std=0.1)
nn.init.normal(self.recon_2.weight, mean=0,std=0.1)
nn.init.normal(self.out_recon.weight, mean=0,std=0.1)
"""
