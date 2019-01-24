'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

#from keras import layers, models
#from keras import backend as K
#K.set_image_data_format('channels_last')

import torch
import torch.nn as nn
import torch.nn.functional as F

from capsule_layers_pytorch import Mask, Length #DeconvCapsuleLayer

import sys
sys.path.append("../DynamicRouting")
from layers import CapsuleLayer, calc_same_padding


class CapsNetR3(nn.Module):
    def __init__(self, n_class=2):
        super(CapsNetR3, self).__init__()
        # Layer 1: Just a conventional Conv2D layer
        d = 512
        padding, d = calc_same_padding(d, kernel=5, stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=padding, bias=True)
        #conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
    
        device = torch.device('cuda')
    
        # Reshape layer to be 1 capsule x [filters] atoms
        #_, H, W, C = conv1.get_shape()
        #conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        padding, d = calc_same_padding(d, kernel=5, stride=2)
        self.primary_caps = CapsuleLayer(output_dim=2, output_atoms=16, num_routing=1,
                                             voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': padding}, device=device)

        # Layer 2: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=1)
        self.conv_cap_2_1 = CapsuleLayer(output_dim=4, output_atoms=16, num_routing=3,
                                             voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 2: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=2)
        self.conv_cap_2_2 = CapsuleLayer(output_dim=4, output_atoms=32, num_routing=3,
                                             voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 3: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=1)
        self.conv_cap_3_1 = CapsuleLayer(output_dim=8, output_atoms=32, num_routing=3,
                                             voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 3: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=2)
        self.conv_cap_3_2 = CapsuleLayer(output_dim=8, output_atoms=64, num_routing=3,
                                             voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 4: Convolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=1)
        self.conv_cap_4_1 = CapsuleLayer(output_dim=8, output_atoms=32, num_routing=3,
                                             voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 1 Up: Deconvolutional Capsule
        padding, d = calc_same_padding(d, kernel=4, stride=2, transposed=True)
        self.deconv_cap_1_1 = CapsuleLayer(output_dim=8, output_atoms=32, num_routing=3,
                                               voting={'type': 'ConvTranspose2d', 'stride': 2, 'kernel_size': 4, 'padding': padding}, device=device)
    
        # Skip connection
        #up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])
    
        # Layer 1 Up: Deconvolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=1)
        self.deconv_cap_1_2 = CapsuleLayer(output_dim=4, output_atoms=32, num_routing=3,
                                               voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 2 Up: Deconvolutional Capsule
        padding, d = calc_same_padding(d, kernel=4, stride=2, transposed=True)
        self.deconv_cap_2_1 = CapsuleLayer(output_dim=4, output_atoms=16, num_routing=3,
                                               voting={'type': 'ConvTranspose2d', 'stride': 2, 'kernel_size': 4, 'padding': padding}, device=device)
    
        # Skip connection
        #up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])
    
        # Layer 2 Up: Deconvolutional Capsule
        padding, d = calc_same_padding(d, kernel=5, stride=1)
        self.deconv_cap_2_2 = CapsuleLayer(output_dim=4, output_atoms=16, num_routing=3,
                                               voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 5, 'padding': padding}, device=device)
    
        # Layer 3 Up: Deconvolutional Capsule
        padding, d = calc_same_padding(d, kernel=4, stride=2, transposed=True)
        self.deconv_cap_3_1 = CapsuleLayer(output_dim=2, output_atoms=16, num_routing=3,
                                               voting={'type': 'ConvTranspose2d', 'stride': 2, 'kernel_size': 4, 'padding': padding}, device=device)
    
        # Skip connection
        #up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])
    
        # Layer 4: Convolutional Capsule: 1x1
        padding, d = calc_same_padding(d, kernel=1, stride=1)
        self.seg_caps = CapsuleLayer(output_dim=1, output_atoms=16, num_routing=3,
                                         voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 1, 'padding': padding}, device=device)
    
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        self.out_seg = Length(num_classes=n_class, seg=True)
    
        # Decoder network.
        #_, H, W, C, A = seg_caps.get_shape()
        #y = layers.Input(shape=input_shape[:-1]+(1,))
        self.mask = Mask() # The true label is used to mask the output of capsule layer. For training
        #masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction
    
            # Decoder
        self.recon_1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)

        self.recon_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)

        self.out_recon = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, y): # y is the true label
        # Layer 1: Just a conventional Conv2D layer
        conv1 = self.conv1(x)
        #conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
    
        # Reshape layer to be 1 capsule x [filters] atoms
        #_, H, W, C = conv1.get_shape()
        conv1_reshaped = conv1.unsqueeze(1)
        #conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = self.primary_caps(conv1_reshaped)
        #primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same', routings=1, name='primarycaps')(conv1_reshaped)
    
        # Layer 2: Convolutional Capsule
        conv_cap_2_1 = self.conv_cap_2_1(primary_caps)
        #conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same', routings=3, name='conv_cap_2_1')(primary_caps)
    
        # Layer 2: Convolutional Capsule
        conv_cap_2_2 = self.conv_cap_2_2(conv_cap_2_1)
        #conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=2, padding='same', routings=3, name='conv_cap_2_2')(conv_cap_2_1)
    
        # Layer 3: Convolutional Capsule
        conv_cap_3_1 = self.conv_cap_3_1(conv_cap_2_2)
        #conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same', routings=3, name='conv_cap_3_1')(conv_cap_2_2)
    
        # Layer 3: Convolutional Capsule
        conv_cap_3_2 = self.conv_cap_3_2(conv_cap_3_1)
        #conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64, strides=2, padding='same', routings=3, name='conv_cap_3_2')(conv_cap_3_1)
    
        # Layer 4: Convolutional Capsule
        conv_cap_4_1 = self.conv_cap_4_1(conv_cap_3_2)
        #conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=1, padding='same', routings=3, name='conv_cap_4_1')(conv_cap_3_2)
    
        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_1 = self.deconv_cap_1_1(conv_cap_4_1)
        #deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=32, upsamp_type='deconv', scaling=2, padding='same', routings=3, name='deconv_cap_1_1')(conv_cap_4_1)
    
        # Skip connection
        up_1 = torch.cat([deconv_cap_1_1, conv_cap_3_1], dim=1)
        #up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])
    
        # Layer 1 Up: Deconvolutional Capsule
        deconv_cap_1_2 = self.deconv_cap_1_2(up_1)
        #deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32, strides=1, padding='same', routings=3, name='deconv_cap_1_2')(up_1)
    
        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_1 = self.deconv_cap_2_1(deconv_cap_1_2)
        #deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=16, upsamp_type='deconv', scaling=2, padding='same', routings=3, name='deconv_cap_2_1')(deconv_cap_1_2)
    
        # Skip connection
        up_2 = torch.cat([deconv_cap_2_1, conv_cap_2_1], dim=1)
        #up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])
    
        # Layer 2 Up: Deconvolutional Capsule
        deconv_cap_2_2 = self.deconv_cap_2_2(up_2)
        #deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=16, strides=1, padding='same', routings=3, name='deconv_cap_2_2')(up_2)
    
        # Layer 3 Up: Deconvolutional Capsule
        deconv_cap_3_1 = self.deconv_cap_3_1(deconv_cap_2_2)
        #deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=16, upsamp_type='deconv', scaling=2, padding='same', routings=3, name='deconv_cap_3_1')(deconv_cap_2_2)
    
        # Skip connection
        up_3 = torch.cat([deconv_cap_3_1, conv1_reshaped], dim=1)
        #up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])
    
        # Layer 4: Convolutional Capsule: 1x1
        seg_caps = self.seg_caps(up_3)
        #seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16, strides=1, padding='same', routings=3, name='seg_caps')(up_3)
    
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = self.out_seg(seg_caps)
    
        # Decoder network.
        _, C, A, H, W = seg_caps.shape
        #y = x[:-1]+(1,)
        masked_by_y = self.mask([seg_caps, y]) # -> (1,512,512,1,16)     The true label is used to mask the output of capsule layer. For training
        #masked = self.mask(seg_caps)  # Mask using the capsule with maximal length. For prediction
    
    
        #def shared_decoder(mask_layer):
        recon_remove_dim = masked_by_y.view(-1, A, H, W) # -> (1,512,512,16)

        recon_1 = F.relu(self.recon_1(recon_remove_dim))# -> (1,512,512,64)

        recon_2 = F.relu(self.recon_2(recon_1))# -> (1,512,512,128)

        out_recon = torch.sigmoid(self.out_recon(recon_2))# -> (1,512,512,1)

        return out_seg, out_recon

"""    
nn.init.normal(self.recon_1.weight, mean=0,std=0.1)
nn.init.normal(self.recon_2.weight, mean=0,std=0.1)
nn.init.normal(self.out_recon.weight, mean=0,std=0.1)
"""

class CapsNetBasic(nn.Module):
    def __init__(self, n_class=2):
        super(CapsNetBasic, self).__init__()

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True) # -> (1,512,512,256
        #conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1')(x)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        #self.primary_caps = ConvCapsuleLayer(kernel_size=5, input_num_capsule=1, input_num_atoms=256, num_capsule=8, num_atoms=32, strides=1, padding=2, routings=1)
        self.primary_caps = ConvCapsuleLayer(output_dim=32, input_atoms=256, output_atoms=8, num_routing=1, stride=1, kernel_size=5, padding=2, use_cuda=True)
    
        # Layer 4: Convolutional Capsule: 1x1
        #self.seg_caps = ConvCapsuleLayer(kernel_size=1, input_num_capsule=8, input_num_atoms=32, num_capsule=1, num_atoms=16, strides=1, padding=0, routings=3)
        self.seg_caps = ConvCapsuleLayer(output_dim=1, input_atoms=8, output_atoms=16, num_routing=3, stride=1, kernel_size=1, padding=0, use_cuda=True)
    
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        self.out_seg = Length(num_classes=n_class, seg=True)
        # Decoder network.
        self.mask = Mask()  # The true label is used to mask the output of capsule layer. For training
        #masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction
    
        # Decoder
        self.recon_1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True)

        self.recon_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)

        self.out_recon = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x, y): # y is the true label

        # Layer 1: Just a conventional Conv2D layer
        conv1 = F.relu(self.conv1(x)) # -> (1,512,512,256)

        # Reshape layer to be 1 capsule x [filters] atoms
        conv1_reshaped = conv1.unsqueeze(1)
    
        # Layer 1: Primary Capsule: Conv cap with routing 1
        primary_caps = self.primary_caps(conv1_reshaped) # -> (1,512,512,8,32)
    
        # Layer 4: Convolutional Capsule: 1x1
        seg_caps = self.seg_caps(primary_caps) # -> (1,512,512,1,16)
    
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        out_seg = self.out_seg(seg_caps) # -> (1, 512,512,1)
    
        # Decoder network.
        _, C, A, H, W = seg_caps.shape
        #y = x[:-1]+(1,)
        masked_by_y = self.mask([seg_caps, y]) # -> (1,512,512,1,16)     The true label is used to mask the output of capsule layer. For training
        #masked = self.mask(seg_caps)  # Mask using the capsule with maximal length. For prediction
    
    
        #def shared_decoder(mask_layer):
        recon_remove_dim = masked_by_y.view(-1, A, H, W) # -> (1,512,512,16)

        recon_1 = F.relu(self.recon_1(recon_remove_dim))# -> (1,512,512,64)

        recon_2 = F.relu(self.recon_2(recon_1))# -> (1,512,512,128)

        out_recon = torch.sigmoid(self.out_recon(recon_2))# -> (1,512,512,1)

        return out_seg, out_recon
