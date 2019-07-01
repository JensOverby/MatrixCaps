'''
Created on Jan 14, 2019

@author: jens
'''
import torch
import torch.nn as nn
from collections import OrderedDict
import layers
import layers2
import batchrenorm as af

class MSELossWeighted(nn.Module):
    def __init__(self, batch_size=1, transition_loss=0., weight=None, weight2=None, pretrained=False):
        super(MSELossWeighted, self).__init__()
        if weight is None:
            weight = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]).cuda()
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

    """
        if self.normalize > 0:
            lenx = p[:,:3].norm(dim=1, p=2, keepdim=True)
            leny = p[:,3:6].norm(dim=1, p=2, keepdim=True)
            A, B = (0.2-1)/0.2, 1
            lenx = torch.where(lenx > 0.2, lenx, A*lenx+B)
            leny = torch.where(leny > 0.2, leny, A*leny+B)
            pnorm = p.new_full(p.size(), 1)
            pnorm[:,:3] = lenx
            pnorm[:,3:6] = leny
            p = p / pnorm

            if self.normalize > 1:
                len_b = p[:,3:6].norm(dim=1, p=2, keepdim=True)
                a = p.data[:,:3]
                b = p.data[:,3:6]
                c = torch.cross(a,b,dim=1)
                b1 = torch.cross(c,a,dim=1)
                b1 = b1 / b1.norm(dim=1,keepdim=True,p=2)
                b1 = b1 * len_b
                pnorm[:,3:6] = b1/b
                p = p * pnorm
    """

def insert(ordereddict, key, newkey, object):
    new_orderded_dict=ordereddict.__class__()
    for i, value in ordereddict.items():
        new_orderded_dict[i]=value
        if i==key:
            new_orderded_dict[newkey]=object
    ordereddict.clear()
    ordereddict.update(new_orderded_dict)


class CapsNet(nn.Module):
    def __init__(self, args, stat=None):
        super(CapsNet, self).__init__()

        self.recon_factor = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.regularize_factor = nn.Parameter(torch.tensor(1e-6), requires_grad=False)
        self.routing_list = []

        if args.dataset == 'rabbit200x100':
            """
            OBS: primary caps 2 and 3 should be WITHOUT BIAS!! Bias is NOT good...!
            """
            layer_list = OrderedDict()
            right_container = []
            layer_list['split_stereo'] = layers2.SplitStereoReturnLeftLayer(right_container)
            
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=3+1, out_channels=10, kernel_size=15, stride=1, padding=7, bias=False)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            #nn.init.normal_(layer_list['conv1'].bias.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=10, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=8, h=9, kernel_size=15, stride=2, padding=7, bias=True)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=8, num_routing=1)

            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=8, h=9, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=8, num_routing=3)

            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=32, h=14, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=32, num_routing=3)


            left_container = []
            layer_list['store_left'] = layers2.StoreLayer(left_container, False)
            layer_list['activate_right'] = layers2.ActivatePathway(right_container)
            layer_list['right_posenc'] = layer_list['posenc']
            layer_list['right_conv1'] = layer_list['conv1']
            layer_list['right_bn1'] = layer_list['bn1']
            layer_list['right_relu1'] = layer_list['relu1']
            layer_list['right_prim1'] = layer_list['prim1']
            layer_list['right_bnn1'] = layer_list['bnn1']
            layer_list['right_route1'] = layer_list['route1']
            layer_list['right_prim2'] = layer_list['prim2']
            layer_list['right_bnn2'] = layer_list['bnn2']
            layer_list['right_route2'] = layer_list['route2'] 
            layer_list['right_prim3'] = layer_list['prim3']
            layer_list['right_bnn3'] = layer_list['bnn3']
            layer_list['right_route3'] = layer_list['route3']
            layer_list['concat'] = layers2.ConcatLayer(left_container, do_clone=False)


            self.decoder_input_atoms = 10
            layer_list['prim4'] = layers.PrimMatrix2d(output_dim=1, h=self.decoder_input_atoms, kernel_size=0, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn4'] = layers2.BNLayer()
            layer_list['route4'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            self.capsules = nn.Sequential(layer_list)
            self.image_decoder = None

        elif args.dataset == 'rabbit100x100':
            """
            OBS: primary caps 2 and 3 should be WITHOUT BIAS!! Bias is NOT good...!
            """

            layer_list = OrderedDict()
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=3+1, out_channels=17, kernel_size=15, stride=1, padding=7, bias=False)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=17, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=8, h=16, kernel_size=15, stride=2, padding=7, bias=True)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=8, num_routing=1)

            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=8, h=16, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=8, num_routing=3)

            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=32, h=16, kernel_size=9, stride=2, padding=4, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=32, num_routing=3)

            self.decoder_input_atoms = 10
            layer_list['prim4'] = layers.PrimMatrix2d(output_dim=1, h=self.decoder_input_atoms, kernel_size=0, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn4'] = layers2.BNLayer()
            layer_list['route4'] = layers.MatrixRouting(output_dim=1, num_routing=3)
            self.capsules = nn.Sequential(layer_list)

            decoder_list = OrderedDict()
            #decoder_list['prepare'] = layers2.Pose2VectorRepLayer()
            decoder_list['1transposed'] = layers.PrimMatrix2d(output_dim=16, h=16, kernel_size=9, stride=1, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn1_transposed'] = layers2.BNLayer()
            decoder_list['route1_transposed'] = layers.MatrixRouting(output_dim=16, num_routing=3)

            decoder_list['2transposed'] = layers.PrimMatrix2d(output_dim=16, h=16, kernel_size=9, stride=2, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn2_transposed'] = layers2.BNLayer()
            decoder_list['route2_transposed'] = layers.MatrixRouting(output_dim=16, num_routing=3)

            decoder_list['3transposed'] = layers.PrimMatrix2d(output_dim=8, h=16, kernel_size=9, stride=2, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn3_transposed'] = layers2.BNLayer()
            decoder_list['route3_transposed'] = layers.MatrixRouting(output_dim=8, num_routing=3)

            decoder_list['transform'] = layers.MatrixToConv()

            decoder_list['conv1_transposed'] = nn.ConvTranspose2d(in_channels=8*17, out_channels=3, kernel_size=7, stride=2, padding=10, output_padding=1, bias=True)
            nn.init.normal_(decoder_list['conv1_transposed'].weight.data, mean=0,std=0.1)

            self.image_decoder = nn.Sequential(decoder_list)

        elif args.dataset == 'MNIST':

            A, B, C, D, E, h = 32, 32, 32, 32, 10, 16
            img_size = 784
            
            layer_list = OrderedDict()
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=2, out_channels=A, kernel_size=5, stride=2, padding=0, bias=False)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            #nn.init.normal_(layer_list['conv1'].bias.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=A, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)
    
            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=B, h=16, kernel_size=1, stride=1, padding=0, bias=True, advanced=False)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=B, num_routing=1, experimental=False, sparse=None)
    
            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=C, h=16, kernel_size=3, stride=2, padding=0, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=C, num_routing=3, experimental=False) #, sparse=layers.SparseCoding(C, return_mask=True))
            #layer_list['boost2'] = layers.Boost()
    
            layer_list['prim2a'] = layers.PrimMatrix2d(output_dim=D, h=16, kernel_size=3, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn2a'] = layers2.BNLayer()
            layer_list['route2a'] = layers.MatrixRouting(output_dim=D, num_routing=3, experimental=False) #, sparse=layers.SparseCoding(D, return_mask=True))
            #layer_list['boost2a'] = layers.Boost()
    
            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=E, h=16, kernel_size=0, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            #layer_list['route3'] = layers.MatrixRouting(output_dim=E, num_routing=3, experimental=False)
            route3 = layers.MatrixRouting(output_dim=E, num_routing=3, experimental=True, sparse=layers.SparseCoding(E, return_mask=False), stat=stat)
            self.routing_list.append(route3)
            layer_list['route3'] = route3
            #layer_list['boost3'] = layers.Boost()
            layer_list['cat'] = layers2.CatLayer()

            self.capsules = nn.Sequential(layer_list)

            self.image_decoder = nn.Sequential(
                layers2.MaskLayer(-1, one_hot=False),
                nn.Linear((h+1) * E, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, img_size),
                nn.Sigmoid()
            )

        elif args.dataset == 'MNIST_ORIGINAL':

            A, B, C, D, E, h = 32, 32, 32, 32, 10, 4
            
            layer_list = OrderedDict()
            layer_list['conv1'] = nn.Conv2d(in_channels=1, out_channels=A, kernel_size=5, stride=2, padding=0, bias=True)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            layer_list['relu1'] = nn.ReLU(inplace=True)
    
            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=B, h=16, kernel_size=1, stride=1, padding=0, bias=True, advanced=False)
            layer_list['sigmoid1'] = layers2.SigmoidLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=B, num_routing=1)
    
    
            layer_list['caps2'] = layers.ConvMatrix2d(output_dim=C, hh=16, kernel_size=3, stride=2)
            layer_list['route2'] = layers.MatrixRouting(output_dim=C, num_routing=3)

            layer_list['caps3'] = layers.ConvMatrix2d(output_dim=D, hh=16, kernel_size=3, stride=1)
            layer_list['route3'] = layers.MatrixRouting(output_dim=C, num_routing=3)

            layer_list['caps4'] = layers.ConvMatrix2d(output_dim=E, hh=16, kernel_size=0, stride=1)
            layer_list['route4'] = layers.MatrixRouting(output_dim=E, num_routing=3)

            self.capsules = nn.Sequential(layer_list)

            self.image_decoder = nn.Sequential(
                layers2.MaskLayer(E),
                nn.Linear(h*h * E, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 784),
                nn.Sigmoid()
            )
            
        elif args.dataset == 'smallNORB':

            A, B, C, D, E, h = 17, 16, 32, 32, 5, 16
            img_size = 32*32

            layer_list = OrderedDict()
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=1+1, out_channels=A, kernel_size=9, stride=1, padding=2, bias=False)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=A, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=B, h=h, kernel_size=9, stride=2, padding=2, bias=True)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=B, num_routing=1)

            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=C, h=h, kernel_size=7, stride=2, padding=1, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=C, num_routing=3)

            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=D, h=h, kernel_size=5, stride=2, padding=1, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=D, num_routing=3)

            #self.decoder_input_atoms = 16
            layer_list['prim4'] = layers.PrimMatrix2d(output_dim=E, h=h, kernel_size=0, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn4'] = layers2.BNLayer()
            layer_list['route4'] = layers.MatrixRouting(output_dim=E, num_routing=3)
            self.capsules = nn.Sequential(layer_list)

            self.capsules = nn.Sequential(layer_list)

            self.image_decoder = nn.Sequential(
                layers2.MaskLayer(E),
                nn.Linear(h * E, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, img_size),
                nn.Sigmoid()
            )
        elif args.dataset == 'msra':
            """
            OBS: primary caps 2 and 3 should be WITHOUT BIAS!! Bias is NOT good...!
            """
            A, B, C, D, E, F, h = 13, 16, 32, 32, 32, 5, 12
            
            layer_list = OrderedDict()
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=1+1, out_channels=A, kernel_size=15, stride=1, padding=0, bias=False)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=A, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=B, h=h, kernel_size=13, stride=2, padding=0, bias=True) # -> 51
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=B, num_routing=1)
            #layer_list['route1'] = layers.MatrixRouting(output_dim=B, num_routing=1, activation=af.NormedActivation(B, momentum=0.1*(args.batch_size/20), update_interval=1, activation=af.Sigmoid()))

            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=C, h=h, kernel_size=9, stride=2, padding=1, bias=False, advanced=True) # -> 23
            layer_list['bnn2'] = layers2.BNLayer()
            route2 = layers.MatrixRouting(output_dim=C, num_routing=3, activation=af.NormedActivation(C, momentum=0.1*(args.batch_size/20), update_interval=3,
                                            activation=af.Sigmoid()), sparse=layers.SparseCoding(C, type='lifetime',
                                            target_max_boost=1.5, return_mask=False, active=True), stat=stat)
            #route2 = layers.MatrixRouting(output_dim=C, num_routing=3, experimental=True, sparse=layers.SparseCoding(C, type='population', down_boost_factor=0.5, up_boost_factor=0.05, target_max_boost=0.6, target_min_boost=0.47, return_mask=False, active=True), stat=stat)
            #self.routing_list.append(route2)
            layer_list['route2'] = route2
            
            layer_list['prim2a'] = layers.PrimMatrix2d(output_dim=D, h=h, kernel_size=7, stride=2, padding=0, bias=False, advanced=True) # -> 9
            layer_list['bnn2a'] = layers2.BNLayer()
            route2a = layers.MatrixRouting(output_dim=D, num_routing=3, activation=af.NormedActivation(D, momentum=0.1*(args.batch_size/20), update_interval=3,
                                            activation=af.Sigmoid()), sparse=layers.SparseCoding(D, type='lifetime',
                                            target_max_boost=1.5, return_mask=False, active=True), stat=stat)
            #route2a = layers.MatrixRouting(output_dim=C, num_routing=3, experimental=True, sparse=layers.SparseCoding(C, type='population', down_boost_factor=0.5, up_boost_factor=0.05, target_max_boost=0.6, target_min_boost=0.47, return_mask=False, active=True), stat=stat)
            #self.routing_list.append(route2a)
            layer_list['route2a'] = route2a
            #0.996
            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=E, h=h, kernel_size=5, stride=1, padding=0, bias=False, advanced=True) # -> 5
            layer_list['bnn3'] = layers2.BNLayer()
            route3 = layers.MatrixRouting(output_dim=E, num_routing=3, activation=af.NormedActivation(E, momentum=0.1*(args.batch_size/20), update_interval=3,
                                            activation=af.Sigmoid()), sparse=layers.SparseCoding(E, type='lifetime',
                                            target_max_boost=1.5, return_mask=False, active=True), stat=stat)
            #route3 = layers.MatrixRouting(output_dim=D, num_routing=3, experimental=True, sigmoid_offset=0., sparse=layers.SparseCoding(D, type='population', down_boost_factor=1., up_boost_factor=0.1, target_max_boost=0.6, target_min_boost=0.47, return_mask=False), stat=stat)
            #self.routing_list.append(route3)
            layer_list['route3'] = route3

            container = []
            layer_list['store'] = layers2.StoreLayer(container)

            self.decoder_input_atoms = 15
            layer_list['prim4'] = layers.PrimMatrix2d(output_dim=F, h=self.decoder_input_atoms, kernel_size=0, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn4'] = layers2.BNLayer()
            layer_list['route4'] = layers.MatrixRouting(output_dim=F, num_routing=3)
            #layer_list['route4'] = layers.MatrixRouting(output_dim=E, num_routing=3, activation=af.NormedActivation(E, momentum=0.1*(args.batch_size/20), update_interval=3, activation=af.Sigmoid()))

            layer_list['cat'] = layers2.CatLayer()
            
            self.capsules = nn.Sequential(layer_list)

            """
            decoder_list = OrderedDict()
            
            decoder_list['activate'] = layers2.ActivatePathway(container)
            #decoder_list['mask'] = layers2.MaskLayer()
            
            #decoder_list['prepare'] = layers2.Pose2VectorRepLayer()
            decoder_list['1transposed'] = layers.PrimMatrix2d(output_dim=32, h=15, kernel_size=10, stride=1, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn1_transposed'] = layers2.BNLayer()
            decoder_list['route1_transposed'] = layers.MatrixRouting(output_dim=32, num_routing=3)
            #decoder_list['sparse1_transposed'] = sparse.SparseCoding()
    
            decoder_list['2transposed'] = layers.PrimMatrix2d(output_dim=16, h=12, kernel_size=7, stride=1, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn2_transposed'] = layers2.BNLayer()
            decoder_list['route2_transposed'] = layers.MatrixRouting(output_dim=16, num_routing=3)
            #decoder_list['sparse2_transposed'] = sparse.SparseCoding()
    
            decoder_list['3transposed'] = layers.PrimMatrix2d(output_dim=1, h=12, kernel_size=7, stride=2, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn3_transposed'] = layers2.BNLayer()
            decoder_list['route3_transposed'] = layers.MatrixRouting(output_dim=1, num_routing=3)
    
            decoder_list['transform'] = layers.MatrixToConv()
    
            decoder_list['conv1_transposed'] = nn.ConvTranspose2d(in_channels=13, out_channels=1, kernel_size=11, stride=2, padding=0, output_padding=1, bias=True)
            nn.init.normal_(decoder_list['conv1_transposed'].weight.data, mean=0,std=0.1)
            """
            self.image_decoder = None #nn.Sequential(decoder_list)
            
        """
        self.route_list = []
        
        for name, module in self.capsules.named_modules():
            if name[0:5] == 'route' and name[-6:] != 'sparse':
                sz = module.output_dim
                rmodule = module
                dist = (torch.rand(10000,sz) > 2/3).float()
                mu = dist.mean()
                var = ((dist - mu) ** 2).sum() / dist.numel()
                self.route_list.append((module, mu, var))
        #module = self.route_list[-1][0]
        #self.route_list.pop(-1)
        dist = (torch.rand(10000,sz) > (sz-1)/sz).float()
        mu = dist.mean()
        var = ((dist - mu) ** 2).sum() / dist.numel()
        self.route_list.append((rmodule, mu, var))
        elif dataset == 'msra':
            A, B, C, D, E, h = 13, 16, 32, 32, 5, 12
            
            layer_list = OrderedDict()
            layer_list['posenc'] = layers.PosEncoderLayer()
            layer_list['conv1'] = nn.Conv2d(in_channels=1+1, out_channels=A, kernel_size=15, stride=3, padding=0, bias=False)
            nn.init.normal_(layer_list['conv1'].weight.data, mean=0,std=0.1)
            layer_list['bn1'] = nn.BatchNorm2d(num_features=A, eps=0.001, momentum=0.1, affine=True)
            layer_list['relu1'] = nn.ReLU(inplace=True)

            layer_list['prim1'] = layers.PrimMatrix2d(output_dim=B, h=h, kernel_size=11, stride=2, padding=0, bias=True)
            layer_list['bnn1'] = layers2.BNLayer()
            layer_list['route1'] = layers.MatrixRouting(output_dim=B, num_routing=1)

            layer_list['prim2'] = layers.PrimMatrix2d(output_dim=C, h=h, kernel_size=7, stride=2, padding=0, bias=False, advanced=True)
            layer_list['bnn2'] = layers2.BNLayer()
            layer_list['route2'] = layers.MatrixRouting(output_dim=C, num_routing=3)

            layer_list['prim3'] = layers.PrimMatrix2d(output_dim=D, h=h, kernel_size=5, stride=2, padding=0, bias=False, advanced=True)
            layer_list['bnn3'] = layers2.BNLayer()
            layer_list['route3'] = layers.MatrixRouting(output_dim=D, num_routing=3)

            self.decoder_input_atoms = 15
            layer_list['prim4'] = layers.PrimMatrix2d(output_dim=E, h=self.decoder_input_atoms, kernel_size=0, stride=1, padding=0, bias=False, advanced=True)
            layer_list['bnn4'] = layers2.BNLayer()
            layer_list['route4'] = layers.MatrixRouting(output_dim=E, num_routing=3)
            self.capsules = nn.Sequential(layer_list)

            decoder_list = OrderedDict()
            #decoder_list['prepare'] = layers2.Pose2VectorRepLayer()
            decoder_list['1transposed'] = layers.PrimMatrix2d(output_dim=32, h=15, kernel_size=10, stride=1, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn1_transposed'] = layers2.BNLayer()
            decoder_list['route1_transposed'] = layers.MatrixRouting(output_dim=32, num_routing=3)
    
            decoder_list['2transposed'] = layers.PrimMatrix2d(output_dim=16, h=12, kernel_size=5, stride=2, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn2_transposed'] = layers2.BNLayer()
            decoder_list['route2_transposed'] = layers.MatrixRouting(output_dim=16, num_routing=3)
    
            decoder_list['3transposed'] = layers.PrimMatrix2d(output_dim=1, h=12, kernel_size=7, stride=2, padding=0, bias=False, advanced=True, func='ConvTranspose2d')
            decoder_list['bnn3_transposed'] = layers2.BNLayer()
            decoder_list['route3_transposed'] = layers.MatrixRouting(output_dim=1, num_routing=3)
    
            decoder_list['transform'] = layers.MatrixToConv()
    
            decoder_list['conv1_transposed'] = nn.ConvTranspose2d(in_channels=13, out_channels=1, kernel_size=11, stride=2, padding=6, output_padding=1, bias=True)
            nn.init.normal_(decoder_list['conv1_transposed'].weight.data, mean=0,std=0.1)
    
            self.image_decoder = nn.Sequential(decoder_list)
        """
        
    def forward(self, x, disable_recon=False):
        p = self.capsules(x)
        if not disable_recon and self.image_decoder is not None:
            return p, self.image_decoder(p)
        return p
