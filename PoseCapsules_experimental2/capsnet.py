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
import layers2
import math
from batchrenorm import BatchRenorm
from sparse import SparseCoding

class MSELossWeighted(nn.Module):
    def __init__(self, batch_size=1, transition_loss=0., weight=None, weight2=None, pretrained=False):
        super(MSELossWeighted, self).__init__()
        if weight is None:
            weight = torch.tensor([4.,4.,4.,4.,4.,4.,1.,1.,1.,1.]).cuda()
            weight = 10. * weight / weight.sum()
        if weight2 is None:
            weight2 = torch.tensor([1.,1.,1.,1.,1.,1.,5.,5.,5.,1.]).cuda()
        
        self.weight = weight
        self.transition_loss = transition_loss
        self.batch_size = batch_size
        self.weight2 = weight2
        self.pretrained = pretrained
        self.trans = (weight2 - weight) / 300.
        self.count = -1
        
        
    def forward(self, input, target):
        pct_var = input-target
        out = (pct_var * self.weight.expand_as(target)) ** 2
        loss = out.sum() 
        if self.count < 300:
            if self.count > -1:
                self.count += 1
                self.weight = self.weight + self.trans
            else:
                if loss/self.batch_size < self.transition_loss:
                    self.count = 0
                    if self.pretrained:
                        self.weight = self.weight2
                        self.count = 300
        return loss        


def hookFunc(module, gradInput, gradOutput):
    #print(len(gradInput))
    for v in gradInput:
        print('Number of in NANs = ', (v != v).sum())
        #print (v)
    for v in gradOutput:
        print('Number of out NANs = ', (v != v).sum())


class CapsNet(nn.Module):
    def __init__(self, output_atoms, img_shape, dataset, data_rep='MSE', normalize=0):
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
            """
            OBS: primary caps 2 and 3 should be WITHOUT BIAS!! Bias is NOT good...!
            """

            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0]+1, out_channels=10, kernel_size=15, stride=1, padding=7, bias=True)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            nn.init.normal_(layer_list['conv1'].bias.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=10, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=8, h=9, kernel_size=15, stride=2, padding=7, bias=True, activate=True)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=8, num_routing=1)

            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=8, h=9, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=8, num_routing=3)

            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=32, h=14, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=32, num_routing=3)

            decoder_input_atoms = 10
            layer_list['prim4'] = layers.PrimMatrix2d(output_dim=1, h=decoder_input_atoms-1, kernel_size=13, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn4'] = layers2.BNLayer()
            layer_list['route4'] = layers.MatrixRouting(output_dim=1, num_routing=3)

            """
            layer_list['up1'] = layers2.UpsampleLayer(17, pos_embed=True)

            decoder_input_atoms = 10
            layer_list['caps4'] = layers.MatrixCaps(output_dim=1, hh=16) #decoder_input_atoms)
            layer_list['route4c'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            layer_list['out'] = layers.MatrixToOut(decoder_input_atoms)
            """

            self.image_decoder = None



            """
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0]+1, out_channels=10, kernel_size=15, stride=1, padding=7, bias=True)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            nn.init.normal_(layer_list['conv1'].bias.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=10, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=4, h=9, kernel_size=15, stride=1, padding=7, bias=True, activate=True)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=4, num_routing=1)

            pathway1 = []
            layer_list['store_pathway1'] = layers2.StoreLayer(pathway1)
            layer_list['pool'] = layers.MaxActPool(kernel_size=2, stride=2, out_sz=100)
            layer_list['up_ex1'] = layers2.UpsampleLayer(1, pos_embed=True)
            layer_list['caps_ex1'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route_ex1'] = layers.MatrixRouting(output_dim=32, num_routing=3)

            pathway2 = []
            layer_list['store_pathway2'] = layers2.StoreLayer(pathway2)
            layer_list['activate_pathway1'] = layers2.ActivatePathway(pathway1)
            
            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=4, h=9, kernel_size=19, stride=15, padding=9, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=4, num_routing=3)

            layer_list['up1'] = layers2.UpsampleLayer(1, pos_embed=True)

            layer_list['caps1'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route1c'] = layers.MatrixRouting(output_dim=32, num_routing=3)

            layer_list['concat_pathway2'] = layers2.ConcatLayer(pathway2, keep_original=True)
            
            layer_list['caps2'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route2c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            pathway3 = []
            layer_list['store_pathway3'] = layers2.StoreLayer(pathway3)

            layer_list['caps3'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route3c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            layer_list['concat_pathway3'] = layers2.ConcatLayer(pathway3, keep_original=True)
            decoder_input_atoms = 10
            layer_list['caps4'] = layers.MatrixCaps(output_dim=1, hh=16) #decoder_input_atoms)
            layer_list['route4c'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            layer_list['out'] = layers.MatrixToOut(decoder_input_atoms)
            

            self.image_decoder = None
            """


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

        elif img_shape[-1] == 400:

            """ Positional Hierarchical Binary Coding layer """
            #layer_list['posenc'] = layers.PosEncoderLayer()
            
            """ First convolutional layer with stride 1 """
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0]+1, out_channels=10, kernel_size=15, stride=1, padding=7, bias=True)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            nn.init.normal_(layer_list['conv1'].bias.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=10, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            """ Convolutional Capsules Block """
            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=4, h=9, kernel_size=15, stride=2, padding=7, bias=True, activate=True)
            layer_list['bnn1'] = BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=4, num_routing=1)
            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=4, h=9, kernel_size=9, stride=2, padding=4, bias=False)
            layer_list['bnn2'] = BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=4, num_routing=3)
            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=8, h=9, kernel_size=9, stride=2, padding=4, bias=False)
            layer_list['bnn3'] = BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=8, num_routing=3)

            """ Dimensionality Reduction Block """
            layer_list['pool'] = layers.MaxRoutePool(kernel_size=2, stride=2)
            layer_list['maxreduce'] = layers.MaxRouteReduce(out_sz=100)
            layer_list['route_pool'] = layers.MatrixRouting(output_dim=8, num_routing=3)
            
            #layer_list['maxreduce'] = layers.MaxRouteReduce() #, max_pct=75., sum_pct=0.)
            #layer_list['route3a'] = layer_list['route3']

            """ Normal Capsules Block with skip connection """
            layer_list['up1'] = UpsampleLayer(1)
            layer_list['caps1'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route1c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            #pathway3 = []
            #layer_list['store_pathway3'] = StoreLayer(pathway3)
            layer_list['caps2'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route2c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            #layer_list['concat2c'] = ConcatLayer(pathway3, 1, keep_original=True)
            layer_list['caps3'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route3c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            #layer_list['sparse3c'] = SparseCoding(False)
            #layer_list['route3c_again'] = layer_list['route3c']
            pathway4 = []
            layer_list['store_pathway4'] = StoreLayer(pathway4)
            decoder_input_atoms = 10
            layer_list['caps4'] = layers.MatrixCaps(output_dim=1, hh=16) #decoder_input_atoms)
            layer_list['route4c'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            layer_list['out'] = layers.MatrixToOut(decoder_input_atoms)

            self.image_decoder = None


            """
            sz = layers.calc_out(img_shape[-1], kernel=15, stride=2, padding=7)
            same_padding = layers.calc_same_padding(sz, k=7, s=1)
            pathway1 = []
            layer_list['store_pathway1'] = StoreLayer(pathway1, False)
            channels = 4
            layer_list['conv2'] = nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=7, stride=1, padding=same_padding, bias=False)
            layer_list['bn2'] = BatchRenorm(num_features=4)
            layer_list['relu2'] = nn.ReLU(inplace=True)
            layer_list['concat2a'] = ConcatLayer(pathway1, False)
            channels += 4
            layer_list['conv3'] = nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=7, stride=1, padding=same_padding, bias=False)
            layer_list['bn3'] = BatchRenorm(num_features=4)
            layer_list['relu3'] = nn.ReLU(inplace=True)
            layer_list['concat3a'] = ConcatLayer(pathway1, False)
            """
            
        
        
        elif img_shape[-1] == 50:

            """
            OBS: primary caps 2 and 3 should be WITHOUT BIAS!! Bias is NOT good...!
            """

            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=img_shape[0]+1, out_channels=10, kernel_size=15, stride=1, padding=7, bias=True)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            nn.init.normal_(layer_list['conv1'].bias.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=10, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            
            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=4, h=9, kernel_size=15, stride=1, padding=7, bias=True, activate=True)
            layer_list['bnn1'] = layers2.BNLayer()
            #layer_list['prim1'].register_backward_hook(hookFunc)
            layer_list['route1'] = layers.MatrixRouting(output_dim=4, num_routing=1)

            pathway1 = []
            layer_list['store_pathway1'] = layers2.StoreLayer(pathway1)
            layer_list['pool'] = layers.MaxActPool(kernel_size=2, stride=2, out_sz=100)
            #layer_list['rand_ex1'] = RandomizeLayer()
            layer_list['up_ex1'] = layers2.UpsampleLayer(1, pos_embed=False)
            layer_list['caps_ex1'] = layers.MatrixCaps(output_dim=32, hh=16, expand=True)
            layer_list['route_ex1'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            layer_list['caps_ex2'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route_ex2'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            pathway2 = []
            layer_list['store_pathway2'] = layers2.StoreLayer(pathway2)
            layer_list['activate_pathway1'] = layers2.ActivatePathway(pathway1)
            
            #pathway1 = []
            #layer_list['store_pathway1'] = StoreLayer(pathway1)
            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=4, h=9, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=4, num_routing=3)

            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=8, h=9, kernel_size=9, stride=3, padding=4, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=8, num_routing=3)

            #layer_list['maxreduce'] = layers.MaxRouteReduce(route_max_sz=40, route_sum_sz=40, a_max_sz=40, a_sum_sz=40, rnd_sz=0, kernel_size=5, stride=5)

            #layer_list['pool'] = layers.MaxPool(kernel_size=2, stride=2, a_sz=1, r_sz=99, rnd_sz=50)
            #layer_list['route3_again'] = layer_list['route3']

            
            #layer_list['pool'] = layers.MaxRoutePool(kernel_size=2, stride=2)
            #layer_list['maxreduce'] = layers.MaxRouteReduce(out_sz=100)
            #layer_list['route_pool'] = layers.MatrixRouting(output_dim=8, num_routing=3)


            #layer_list['rand1'] = RandomizeLayer()
            layer_list['up1'] = layers2.UpsampleLayer(1)

            """
            layer_list['caps1'] = layers.ConvMatrix2d(output_dim=32, hh=16, kernel_size=3, stride=2)
            layer_list['route1c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            layer_list['caps2'] = layers.ConvMatrix2d(output_dim=32, hh=16, kernel_size=3, stride=2)
            layer_list['route2c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            layer_list['caps3'] = layers.ConvMatrix2d(output_dim=32, hh=16, kernel_size=3, stride=1)
            layer_list['route3c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            layer_list['caps4'] = layers.ConvMatrix2d(output_dim=1, hh=16, kernel_size=3, stride=1)
            layer_list['route4c'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            decoder_input_atoms = 10
            layer_list['out'] = layers.MatrixToOut(decoder_input_atoms)

            """
            
            layer_list['caps1'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route1c'] = layers.MatrixRouting(output_dim=32, num_routing=3)

            layer_list['concat_pathway2'] = layers2.ConcatLayer(pathway2, keep_original=True)
            
            #pathway3 = []
            #layer_list['store_pathway3'] = StoreLayer(pathway3)
            layer_list['caps2'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route2c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            #layer_list['concat2c'] = ConcatLayer(pathway3, 1, keep_original=True)
            layer_list['caps3'] = layers.MatrixCaps(output_dim=32, hh=16)
            layer_list['route3c'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            #layer_list['concat3c'] = ConcatLayer(pathway3, 1, keep_original=True)
            #layer_list['sparse3c'] = SparseCoding(False)
            #layer_list['route3c_again'] = layer_list['route3c']
            #pathway4 = []
            #layer_list['store_pathway4'] = StoreLayer(pathway4)
            decoder_input_atoms = 10
            layer_list['caps4'] = layers.MatrixCaps(output_dim=1, hh=16) #decoder_input_atoms)
            layer_list['route4c'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            layer_list['out'] = layers.MatrixToOut(decoder_input_atoms)
            

            self.image_decoder = None

        self.capsules = nn.Sequential(layer_list)

        if data_rep=='MSE':
            self.target_decoder = nn.BatchNorm1d(num_features=decoder_input_atoms)
        else:
            self.target_decoder = layers.make_decoder( layers.make_decoder_list([decoder_input_atoms, 512, 512, output_atoms], 'tanh') )
            self.target_decoder.add_module('scale_pi', layers.ScaleLayer(math.pi))

        #kaj = list(self.capsules.named_children())
        #kaj.insert(-4, ('route3c_again', kaj[-6][1]))
        #layer_list = OrderedDict(kaj)
        #self.capsules = nn.Sequential(layer_list)
        #self.capsules.maxreduce.use_old=False

    def forward(self, x, disable_recon=False):

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

        if self.image_decoder is None:
            sz = x.shape[-1]
            if sz > 100:
                sz = 100
            self.image_decoder = layers.make_decoder( layers.make_decoder_list([p.shape[1], 512, 2048, 4096, sz**2], 'sigmoid')) #.cuda(layer.device)
            #self.image_decoder.layer1.weight.data[:256,:] *= 50.
            #self.image_decoder.layer2.weight.data *= 1./10.
            #self.image_decoder.layer3.weight.data *= 1./10.

        if not disable_recon: # or self.image_decoder is None:
            reconstructions = self.image_decoder(p)
        else:
            reconstructions = torch.zeros(1)
            
        return p, reconstructions
