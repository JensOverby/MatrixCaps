'''
Capsules for Object Segmentation (SegCaps)
Original Paper: https://arxiv.org/abs/1804.04241
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of the various capsule layers and dynamic routing and squashing functions.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from skimage import transform
from torchvision import transforms
from PIL import Image
from keras.utils.conv_utils import deconv_length

class Length(nn.Module):
    def __init__(self, num_classes, seg=True):
        super(Length, self).__init__()
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg

    def forward(self, inputs):
        if inputs.shape.__len__() == 5:
            assert inputs.shape[1] == 1, 'Error: Must have num_capsules = 1 going into Length'
            inputs = inputs.squeeze(dim=1)
        return inputs.norm(dim=1, keepdim=True)


class Mask(nn.Module):
    def __init__(self, resize_masks=False):
        super(Mask, self).__init__()
        self.resize_masks = resize_masks

    def forward(self, inputs):
        if type(inputs) is list:
            assert len(inputs) == 2
            input, mask = inputs
            _, _, _, hei, wid = input.shape
            if self.resize_masks:
                mask = kaj*transforms.Resize([hei, wid], interpolation = Image.BICUBIC)(mask)
                #mask = transform.resize(mask, (hei.value, wid.value))
                #mask = tf.image.resize_bicubic(mask, (hei.value, wid.value))
            mask = mask.unsqueeze(2)
            if input.shape.__len__() == 3:
                masked = kaj*(mask * input).view(-1)
            else:
                masked = mask * input

        else:
            if kaj*inputs.shape.dim() == 3:
                x = (inputs**2).sum(-1).sqrt()
                
                # PYTORCH one-hot. Can be optimized
                mask = torch.zeros(x[0], x.shape.as_list()[1])
                mask.scatter_(1, x.argmax(1), 1)
                masked = (mask.unsqueeze(-1) * inputs).view(-1)
            else:
                masked = inputs

        return masked


class ConvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, input_num_capsule, input_num_atoms, num_capsule, num_atoms, strides=1, padding=0, routings=3):
        super(ConvCapsuleLayer, self).__init__()
        #self.kernel_size = kernel_size
        self.input_num_capsule = input_num_capsule
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings

        # Transform matrix
        self.W = nn.Parameter(torch.randn(num_capsule * num_atoms, input_num_atoms, kernel_size, kernel_size))
        #self.W = nn.Parameter(torch.randn(self.kernel_size, self.kernel_size, input_num_atoms, num_capsule * num_atoms))

        self.b = nn.Parameter(torch.full((1,
                                           1, num_capsule, num_atoms), 0.1))

        self.built = True

    def forward(self, input_tensor, training=None):
        # input_tensor shape: (1,512,512,1,256)
        #input_transposed = input_tensor.permute(0, 4, 1, 2, 3) # -> (1,1,512,512,256)
        input_shape = input_tensor.shape
        input_tensor_reshaped = input_tensor.view(input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]) # -> (1,512,512,256)

        conv = F.conv2d(input_tensor_reshaped, self.W, stride=self.strides, padding=self.padding) # -> (1,512,512,256)
        #conv = K.conv2d(input_tensor_reshaped, self.W, (self.strides, self.strides),
        #                padding=self.padding, data_format='channels_last')

        #votes_shape = conv.shape
        _, _, conv_height, conv_width = conv.shape # -> _,512,512,_

        votes = conv.view(input_shape[0], input_shape[1], self.num_capsule, self.num_atoms, conv_height, conv_width) # -> (1,1,512,512,8,32)

        # votes: (1,1,8,32,512,512)
        votes = votes.permute(0,1,4,5,2,3)

        logit_shape = torch.Size([input_shape[0], input_shape[1], conv_height, conv_width, self.num_capsule]) # -> (1,1,512,512,8)
        biases_replicated = self.b.repeat(conv_height, conv_width, 1, 1) # -> (512,512,8,32)

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            num_routing=self.routings) # -> (1, 512,512,8,32)

        return activations.permute(0,3,4,1,2)

"""
class DeconvCapsuleLayer(nn.Module):
    def __init__(self, kernel_size, num_capsule, num_atoms, scaling=2, upsamp_type='deconv', padding='same', routings=3):
        super(DeconvCapsuleLayer, self).__init__()
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.scaling = scaling
        self.upsamp_type = upsamp_type
        self.padding = padding
        self.routings = routings

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        if self.upsamp_type == 'subpix':
            self.W = nn.Parameter(torch.randn(self.kernel_size, self.kernel_size, self.input_num_atoms, self.num_capsule * self.num_atoms * self.scaling * self.scaling))
        elif self.upsamp_type == 'resize':
            self.W = nn.Parameter(torch.randn(self.kernel_size, self.kernel_size, self.input_num_atoms, self.num_capsule * self.num_atoms))
        elif self.upsamp_type == 'deconv':
            self.W = nn.Parameter(torch.randn(self.kernel_size, self.kernel_size, self.num_capsule * self.num_atoms, self.input_num_atoms))
        else:
            raise NotImplementedError('Upsampling must be one of: "deconv", "resize", or "subpix"')

        self.conv2d = self.ConvTranspose2d(in_channels=self.input_num_capsule, out_channels=self.num_capsule, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=False)

        self.b = nn.Parameter(torch.full((1, 1, self.num_capsule, self.num_atoms), 0.1))

        self.built = True

    def forward(self, input_tensor, training=None):
        input_transposed = input_tensor.permute(3, 0, 1, 2, 4)
        input_shape = input_transposed.shape
        input_tensor_reshaped = input_transposed.view(input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms)

        if self.upsamp_type == 'resize':
            upsamp = transforms.Resize([self.input_height*self.scaling, self.input_width*self.scaling], interpolation = Image.NEAREST)(input_tensor_reshaped)
            #upsamp = K.resize_images(input_tensor_reshaped, self.scaling, self.scaling, 'channels_last')
            outputs = F.conv2d(upsamp, self.W, stride=1, padding=self.padding)
            #outputs = K.conv2d(upsamp, kernel=self.W, strides=(1, 1), padding=self.padding, data_format='channels_last')
        elif self.upsamp_type == 'subpix':
            conv = F.conv2d(input_tensor_reshaped, self.W, stride=1, padding='same')
            #conv = K.conv2d(input_tensor_reshaped, kernel=self.W, strides=(1, 1), padding='same',
            #                data_format='channels_last')
            outputs = tf.depth_to_space(conv, self.scaling)
        else:
            batch_size = input_shape[1] * input_shape[0]

            # Infer the dynamic output shape:
            out_height = deconv_length(self.input_height, self.scaling, self.kernel_size, self.padding, None)
            out_width = deconv_length(self.input_width, self.scaling, self.kernel_size, self.padding, None)
            output_shape = (batch_size, out_height, out_width, self.num_capsule * self.num_atoms)

            output = F.conv_transpose2d(input_tensor_reshaped, self.W, stride=self.scaling, padding=self.padding, output_padding, self.groups, self.dilation)
            #outputs = K.conv2d_transpose(input_tensor_reshaped, self.W, output_shape, (self.scaling, self.scaling),
            #                         padding=self.padding, data_format='channels_last')

        votes_shape = outputs.shape
        _, conv_height, conv_width, _ = outputs.shape

        votes = outputs.view(input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule, self.num_atoms)

        logit_shape = torch.stack([input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule])
        biases_replicated = self.b.repeat(votes_shape[1], votes_shape[2], 1, 1)

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            num_routing=self.routings)

        return activations
"""
def update_routing(votes, biases, logit_shape, num_dims, input_dim, num_routing):
    # votes: (1,1,512,512,8,32)
    if num_dims == 6:
        votes_t_shape = 5, 0, 1, 2, 3, 4
        r_t_shape = 1, 2, 3, 4, 5, 0
    elif num_dims == 4:
        votes_t_shape = 3, 0, 1, 2
        r_t_shape = 1, 2, 3, 0
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = votes.permute(votes_t_shape) # -> (32,1,1,512,512,8)
    #_, _, _, height, width, caps = votes_trans.shape

    activations = []
    logits = Variable(torch.zeros(logit_shape), requires_grad=False).cuda() # -> (1,1,512,512,8)

    for _ in range(num_routing):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = F.softmax(logits, dim=-1) # -> (1,1,512,512,8)
        preactivate_unrolled = route * votes_trans # -> (32,1,1,512,512,8)
        preact_trans = preactivate_unrolled.permute(r_t_shape) # -> (1,1,512,512,8,32)
        preactivate = preact_trans.sum(dim=1) + biases # -> (1,512,512,8,32)
        activation = _squash(preactivate) # -> (1,512,512,8,32)
        activations.append(activation)
        
        act_3d = activation.data.unsqueeze(1) # -> (1,1,512,512,8,32)
        #-----
        tile_shape = torch.ones(num_dims, dtype=torch.int32).tolist() # -> [1,1,1,1,1,1]
        tile_shape[1] = input_dim
        act_replicated = act_3d.repeat(tile_shape) # -> (1,1,512,512,8,32)
        #-----
        distances = (votes.data * act_replicated).sum(dim=-1) # -> (1,1,512,512,8)
        logits += distances

    return activations[num_routing - 1]

"""
def dynamic_routing(self, votes, b, r):
    batch_size, input_caps, output_caps, output_dim = votes.size()
    route = F.softmax(b)
    preactivation = (route.unsqueeze(2) * votes).sum(dim=1)
    activation = squash(preactivation)

    b_batch = b.expand((batch_size, input_caps, output_caps))
    for _ in range(r):
        activation = activation.unsqueeze(1)
        #-----
        #-----
        b_batch += (votes * activation).sum(-1)
        route = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
        preactivation = (route * votes).sum(dim=1)
        activation = squash(preactivation)

    return activation

def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x
"""

def _squash(input_tensor):
    norm = input_tensor.norm(dim=-1, keepdim=True)
    norm_squared = norm ** 2
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
