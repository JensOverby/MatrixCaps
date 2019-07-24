import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math
#from batchrenorm import BatchRenorm, Sigmoid
import batchrenorm
import util
from torchvision.transforms.functional import affine
#from caffe2.python.embedding_generation_benchmark import device

votes_t_shape = [3, 0, 1, 2, 4, 5]
r_t_shape = [1, 2, 3, 0, 4, 5]
eps = 1e-10
ln_2pi = math.log(2*math.pi)
#lambda_ = 0.0001

def calc_out(input_, kernel=1, stride=1, padding=0, dilation=1):
    return int((input_ + 2*padding - dilation*(kernel-1) - 1) / stride) + 1

def calc_padding(i, k=1, s=1, d=1, dim=2):
    pad = int( (s*(i-1) - i + k + (k-1)*(d-1)) / 2 )
    if (k%2) == 0:
        padding = (pad+1, pad)
    else:
        padding = (pad, pad)
    pad = padding
    for i in range(dim-1):
        pad = pad + padding
    return pad
        
"""
    o = (i + 2*p - k - (k-1)*(d-1)) / s + 1
    s*o = (i + 2*p - k - (k-1)*(d-1)) + s
    s*o - s = i + 2*p - k - (k-1)*(d-1)
    s*o - s - i + k + (k-1)*(d-1) = 2*p
    (s(o-1) - i + k + (k-1)*(d-1)) / 2 = p
"""

def make_decoder(dec_list):
    model = nn.Sequential()
    for i in dec_list:
        model.add_module(*i)
    return model

def make_decoder_list(layer_sizes, out_activation, mean=0.0, std=0.1, bias=0.0):
    last_layer_size = layer_sizes[0]
    layer_sizes.pop(0)
    sz = len(layer_sizes)
    model = []
    for i, layer_size in enumerate(layer_sizes, 1):
        name = 'layer' + str(i)
        layer = nn.Linear(last_layer_size, layer_size)
        nn.init.normal_(layer.weight.data, mean=mean, std=std)
        nn.init.constant_(layer.bias.data, val=bias)
        model.append((name, layer))
        last_layer_size = layer_size
        if i < sz:
            name = 'relu' + str(i)
            model.append((name, nn.ReLU(inplace=True)))
    if out_activation=='sigmoid':
        model.append(('sigm', nn.Sigmoid()))
    elif out_activation=='tanh':
        model.append(('tanh', nn.Tanh()))
    elif out_activation=='relu':
        model.append(('relu', nn.ReLU(inplace=True)))
    else:
        model.append(('softplus', nn.Softplus()))
    return model

class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super(ScaleLayer, self).__init__()
        self.scale = scale
    def forward(self, x):
        return self.scale * x

class MatrixToOut(nn.Module):
    def __init__(self, output_dim):
        super(MatrixToOut, self).__init__()
        self.output_dim = output_dim
    def forward(self, x):
        return x[...,:self.output_dim] #, None, None

class MatrixToConv(nn.Module):
    def __init__(self):
        super(MatrixToConv, self).__init__()
    def forward(self, x):
        y = x[1]
        x = x[0]
        shp = x.shape
        x = x.permute(0,1,4,2,3).contiguous().view(shp[0], -1, shp[2], shp[3])
        y = y.view(shp[0], -1, shp[2], shp[3])
        return torch.cat([x, y], dim=1)

class SinusEncoderLayer(nn.Module):
    """
    Positional Hierarchical Binary Coding
    """
    def __init__(self):
        super(SinusEncoderLayer, self).__init__()
        self.m = None

    def forward(self, x):
        if self.m is None:
            w = x.shape[-1]

            row = torch.zeros(w)
            for i in range(w):
                theta = 2*math.pi*i/w
                row[i] += math.sin(theta)
            row_list = []
            for i in range(w):
                row_list.append(row)
            self.m = torch.stack(row_list)
            self.m = self.m + self.m.transpose(1,0)
            self.m = self.m - self.m.min()
            self.m = self.m / self.m.max()
            
        if x.device != self.m.device:
            self.m = self.m.cuda(x.device)
            
        x = torch.cat([x,self.m[None,None,:,:].repeat(x.shape[0],1,1,1)], dim=1)
        return x

class PosEncoderLayer(nn.Module):
    """
    Positional Hierarchical Binary Coding
    """
    def __init__(self):
        super(PosEncoderLayer, self).__init__()
        self.m = None

    def forward(self, x):
        if self.m is None:
            w = x.shape[-1]
            sz = 1
            while sz < w:
                sz *= 2
            row = torch.zeros(sz)
            
            step = 1
            value = False
            
            while step < sz:
                for i in range(int(sz/step)):
                    row[i*step:(i+1)*step] += int(value)
                    value = not value
                step *= 2
                value = False
            
            row = row[:w]

            row_list = []
            for i in range(w):
                row_list.append(row)
            self.m = torch.stack(row_list)
            self.m = self.m + self.m.transpose(1,0)
            self.m = self.m - self.m.min()
            self.m = self.m / self.m.max()
            self.m = self.m[:x.shape[-2], :]

        if x.device != self.m.device:
            self.m = self.m.cuda(x.device)
            
        x = torch.cat([x,self.m[None,None,:,:].repeat(x.shape[0],1,1,1)], dim=1)
        return x


class ConvVector2d(nn.Module):
    def __init__(self, output_dim, h, kernel_size, stride, padding, bias):
        super(ConvVector2d, self).__init__()
        self.output_dim = output_dim
        self.h = h
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.not_initialized = True
        
    def init(self, x): # batch_size, input_dim, input_atoms, dim_x, dim_y
        self.conv = nn.Conv2d(in_channels=x.size(2),
                                       out_channels=self.output_dim * self.h,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       bias=self.bias)
        nn.init.normal_(self.conv.weight.data, mean=0,std=0.1)
        self.not_initialized = False


    def forward(self, x):
        """ x: batch_size, input_dim, input_atoms, dim_x, dim_y """
        if type(x) is tuple:
            x = x[0]
        shp = x.shape                                           # batch_size, input_dim, input_atoms, dim_x, dim_y

        if len(shp) == 4:
            """ If previous was Conv2d """
            x = x.unsqueeze(1)
            shp = x.shape
        elif (shp[-1] == self.h) and (shp[2] == shp[3]): # b, C, w, w, h
            """ If previous was MatrixRouting """
            x = x.permute(0,1,4,2,3)
            shp = x.shape
        if self.not_initialized:
            self.init(x)
        
        x = x.view(shp[0]*shp[1], shp[2], shp[3], shp[4])  # batch_size*input_dim, input_atoms, dim_x, dim_y
        x = self.conv(x)                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y

        votes = x.view(shp[0], -1, self.output_dim, self.h, x.size(-2), x.size(-1)) # batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
        return votes

"""
def getNormal(sz): 
    y = [] 
    for i in range(sz): 
        b=i*4/sz - 2 
        f=(i+1)*4/sz - 2 
        y.append(norm.cdf(f)-norm.cdf(b)) 
    normal = np.stack(y) 
    b=int((sz-1)/2) 
    f=int(sz/2) 
    remain = (1 - normal.sum())/(f-b+1) 
    for i in range(f-b+1): 
        normal[b+i] += remain 
    rows = normal.reshape(-1,1) 
    cols = normal.reshape(1,-1) 
    kernel = rows*cols 
    return kernel
"""

class PrimMatrix2d(nn.Module):
    def __init__(self, output_dim, h, kernel_size, stride, padding, bias, advanced=False, func='Conv2d', pool=False):
        super(PrimMatrix2d, self).__init__()
        self.output_dim = output_dim
        self.h = h
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.advanced = advanced
        self.not_initialized = True
        self.func = func
        self.pool = pool
        
    def init(self, x): # batch_size, input_dim, input_atoms, dim_x, dim_y
        in_channels = 1
        in_h = x.shape[2]
        ConvFunc = getattr(nn,self.func)
        
        if self.kernel_size == 0:
            self.kernel_size = x.shape[-1]

        if self.advanced:
            same_padding = calc_padding(x.shape[-1], self.kernel_size, s=1, dim=len(x.shape)-3)
            PadFunc = getattr(nn, 'ConstantPad' + self.func[-2:])
            self.pad_pre = PadFunc(same_padding, 0.)
            ConvPreFunc = getattr(nn, 'Conv' + self.func[-2:])
            self.conv_pre = ConvPreFunc(in_channels=in_channels*in_h,
                                           out_channels=in_channels*in_h,
                                           kernel_size=self.kernel_size,
                                           stride=1,
                                           padding=0,
                                           bias=self.bias)
            if x.is_cuda:
                self.conv_pre.cuda()
            nn.init.normal_(self.conv_pre.weight.data, mean=0,std=0.1)
            if self.bias:
                nn.init.normal_(self.conv_pre.bias.data, mean=0,std=0.1)
            in_channels *= 2
            
            #self.hardtanh = nn.Hardtanh(inplace=True)
        
        self.conv = ConvFunc(in_channels=in_channels*in_h,
                                       out_channels=self.output_dim * self.h,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       bias=self.bias)
        if x.is_cuda:
            self.conv.cuda()
        nn.init.normal_(self.conv.weight.data, mean=0,std=0.1)

        if self.bias:
            nn.init.normal_(self.conv.bias.data, mean=0,std=0.1)

        if self.pool:
            self.conv_a = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        else:
            self.conv_a = ConvFunc(in_channels=1,
                                           out_channels=1,
                                           kernel_size=self.kernel_size,
                                           stride=self.stride,
                                           padding=self.padding,
                                           bias=self.bias)
            
            if x.is_cuda:
                self.conv_a.cuda()
            nn.init.normal_(self.conv_a.weight.data, mean=0,std=0.1)
            if self.bias:
                nn.init.normal_(self.conv_a.bias.data, mean=0,std=0.1)
            
        self.not_initialized = False

    def forward(self, x):
        """ x: batch_size, input_dim, input_atoms, dim_x, dim_y """
        if type(x) is tuple:
            y = x[1]
            x = x[0]
            shp = x.shape                                           # batch_size, input_dim, input_atoms, dim_x, dim_y
            if shp[-3] == shp[-2] and shp[-2] != shp[-1]:
                """ If previous was MatrixRouting """
                s = list(range(0, len(shp)-1))
                x = x.permute(s[:2] + [-1] + s[2:])
        else:
            """ Previous was Conv2d """
            x = x.unsqueeze(1)
            y = x[:,:,-1,...]
            x = x[:,:,:-1,...]
        shp = x.shape
            
        if self.not_initialized:
            self.init(x)
        
        x = x.view((shp[0]*shp[1],) + shp[2:])  # batch_size*input_dim, input_atoms, dim_x, dim_y

        if self.advanced:
            votes = self.pad_pre(x)
            votes = self.conv_pre(votes)
            #votes = self.hardtanh(votes)
            votes = F.relu(votes, inplace=True)
            votes = torch.cat([x, votes], dim=1)
            votes = self.conv(votes)                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        else:
            votes = self.conv(x)                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        votes = votes.view((shp[0], -1, self.output_dim, self.h) + votes.shape[2:])

        activations = self.conv_a(y.view(-1, 1, shp[-2], shp[-1]))                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        activations = activations.view((shp[0], -1, 1, 1) + activations.shape[2:]).expand(votes.shape[:3] + (1,) + votes.shape[4:])

        #x = torch.cat([votes, activations], dim=3) # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
        return votes, activations


class ConvMatrix2d(nn.Module):
    def __init__(self, output_dim, hh, kernel_size, stride, padding=0):
        super(ConvMatrix2d, self).__init__()
        self.output_dim = output_dim
        self.h = int(math.sqrt(hh))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.not_initialized = True

    def init(self, x): # batch, input_dim, input_atoms, dim_x, dim_y       (self.b, self.C, self.w, self.w, hh)
        if self.kernel_size == 0:
            self.kernel_size = x.shape[-2]
        self.weight = nn.Parameter(torch.randn(x.shape[1], self.kernel_size, self.kernel_size, self.output_dim, self.h, self.h))  # B,K,K,C,4,4
        self.not_initialized = False

    def forward(self, x): # (self.b, self.C, self.w, self.w, hh)    # batch, input_dim, input_atoms, dim_x, dim_y
        if type(x) is tuple:
            x = x[0]
        shp = x.shape
        
        """ If previous was ConvVector2d """
        if len(shp) == 6: # batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
            x = x.permute(0,1,2,4,5,3)
            x = x.view(shp[0], -1, shp[-2], shp[-1], shp[3])
        if self.not_initialized:
            self.init(x)

        b, input_dim, width_in, _, hh = x.shape
        w = int((width_in - self.kernel_size) / self.stride + 1)
        Bkk = input_dim*self.kernel_size*self.kernel_size
        Cww = self.output_dim*w*w

        pose_list = []
        for j in range(w):
            for i in range(w):
                pose_list.append( x[:, :, self.stride * i:self.stride * i + self.kernel_size, self.stride * j:self.stride * j + self.kernel_size, :] )
        x = torch.stack(pose_list, dim=-2)  # b,B,K,K,w*w,16
        del pose_list

        activations = x[...,hh-1:]
        poses = x[...,:hh-1]
        
        poses = poses.view(b, input_dim, self.kernel_size, self.kernel_size, 1, w, w, self.h, self.h)  # b,B,K,K,1,w,w,4,4
        votes = self.weight[None, :, :, :, :, None, None, :, :] @ poses # b,B,K,K,C,w,w,4,4

        votes = votes.view(b, Bkk, Cww, -1)
        activations = activations.view(b, Bkk, 1, -1).repeat(1, 1, self.output_dim, 1).view(b, Bkk, Cww, 1)
        x = torch.cat([votes, activations], dim=-1)
        
        return x


class ConvCaps(nn.Module):
    
    """
    output_dim:    number of classes
    """
    
    def __init__(self, output_dim, h):
        super(ConvCaps, self).__init__()
        self.not_initialized = True
        self.output_dim = output_dim
        self.h = h

    def init(self, x):
        self.weight = nn.Parameter(torch.Tensor(x.size(1), x.size(2), self.output_dim * self.h, 1, 1))
        nn.init.normal_(self.weight.data, mean=0,std=0.1)      #input_dim, input_atoms, output_dim*h
        self.not_initialized = False

    def forward(self, x):
        if type(x) is tuple:
            x = x[0]
            
        """ If previous was ConvVector2d & VectorRouting """
        if x.shape[-2] > 1:
            x = x.permute(0, 1, 3, 4, 2).contiguous()           # batch_size, input_dim, output_dim, dim_x, dim_y, input_atoms
            x = x.view(x.size(0), -1, x.size(-1), 1, 1)         # batch_size, input_dim*dim_x*dim_y, input_atoms, 1, 1
        if self.not_initialized:
            self.init(x)
        
        x = x.unsqueeze(3)                                         # batch_size, input_dim, input_atoms, 1, dim_x, dim_y
        tile_shape = list(x.size())
        tile_shape[3] = self.output_dim * self.h                # batch_size, input_dim, input_atoms, output_dim*h, dim_x, dim_y
        x = x.expand(tile_shape)                                # batch_size, input_dim, input_atoms, output_dim*h, dim_x, dim_y
        x = torch.sum(x * self.weight, dim=2)                   # batch_size, input_dim, output_dim*h

        votes = x.view(x.size(0), -1, self.output_dim, self.h, x.size(-2), x.size(-1))
        return votes


class MatrixCaps(nn.Module):
    def __init__(self, output_dim, hh):
        super(MatrixCaps, self).__init__()
        self.C = output_dim
        self.h = int(math.sqrt(hh))
        self.not_initialized = True

    def init(self, shp): # b, B, hh
        self.weight = nn.Parameter(torch.randn(shp[1], shp[2], self.C, self.h, self.h))
        self.not_initialized = False

    def forward(self, x): # b, B, hh
        if type(x) is tuple:
            x = x[0]
        shp = x.shape
        if len(shp) == 5:
            if (shp[-2] == shp[-1]) or (shp[-1] == 1):
                x = x.view(shp[0], shp[1], -1)
            else:
                x = x.view(shp[0], shp[1], -1, shp[-1])
                
        if self.not_initialized:
            if (shp[-3] != shp[-2] and shp[-2] != shp[-1]) or (shp[-3] == shp[-2] and shp[-2] != 1):
                x_shp = list(x.shape)
                x_shp[2] = 1
                self.init(x_shp)
            else:
                self.init(x.shape)

        b, B, ww, hh = x.shape

        activations = x[...,hh-1:]
        poses = x[...,:hh-1].view(b, B, ww, 1, self.h, self.h)
        
        votes = self.weight @ poses

        votes = votes.view(b, B*ww, self.C, -1)
        activations = activations.view(b, B*ww, 1, 1).repeat(1, 1, self.C, 1)
        x = torch.cat([votes, activations], dim=-1)
        
        return x


class VectorRouting(nn.Module):
    def __init__(self, num_routing):
        super(VectorRouting, self).__init__()
        self.num_routing = num_routing
        self.bias = None
        
    def init(self, votes): # batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
        if type(votes) is tuple:
            votes = votes[0]
        self.bias = nn.Parameter(torch.Tensor(votes.size(2), votes.size(3), 1, 1))
        nn.init.constant_(self.bias.data, val=0.1)

    def forward(self, votes): # batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
        if self.bias is None:
            self.init(votes)

        """ If previous was MaxRoutePool | MaxRouteReduce """
        if type(votes) is tuple:
            """ the votes are pooled/reduced, so bias is detached """
            votes = votes[0]
            biases_replicated = self.bias.repeat([1,1,votes.size(-2),votes.size(-1)])
        else:
            biases_replicated = self.bias.repeat([1,1,votes.size(-2),votes.size(-1)])
            
        logit_shape = list(votes.size())
        logit_shape.pop(3)                                      # batch_size, input_dim, output_dim, dim_x, dim_y
        x, route = dynamic_routing(votes=votes, biases=biases_replicated, logit_shape=logit_shape, num_routing=self.num_routing)
        return x, route, votes


class SparseCoding(nn.Module):

    def __init__(self, num_features, k=1, type='lifetime', ema_decay=0.95, target_min_boost=0.96, target_max_boost=4., 
                 down_boost_factor=0.05, up_boost_factor=0.05, boost_update_count=30000, sparsity=80/100, return_mask=False, active=True):
        super(SparseCoding, self).__init__()
        self.k = k
        self.masked_freq = False
        self.lifetime = type=='lifetime'
        self.ema_decay=ema_decay
        self.target_min_freq=target_min_boost # Is downscaled in initialization
        self.target_max_freq=target_max_boost # Is downscaled in initialization
        self.down_boost_factor=down_boost_factor
        self.up_boost_factor=up_boost_factor
        self.boost_update_count = boost_update_count
        self.N = boost_update_count - 2
        self.register_buffer('boosting_weights', torch.ones(num_features))
        self.register_buffer('freq_ema', torch.zeros(num_features))
        self.register_buffer('ramp_in', torch.tensor([1.]))
        self.sparsity = sparsity
        self.return_mask = return_mask
        self.active = active
        self.not_initialized = True

        self.steepness_factor=6
        self.clip_threshold=0.02

    """
    batch, dim, dim, input_dim (prim_capsules), output_dim
    
    Find max routing values across all input capsules -> batch,dim,dim,output_dim
    (max of input_dim)
    
    Sum 2nd dim (column), and then sum 1st dim (row) -> batch,output_dim
    """

    def update(self, R):
        """
        x: batch_size, output_dim, h, (dim_x, dim_y)

        Converts the capsule mask into an appropriate shape then applies it to
        the capsule embedding.
    
        Args:
          route: tensor, output of the last capsule layer.
          num_prime_capsules: scalar, number of primary capsules.
          num_latent_capsules: scalar, number of latent capsules.
          verbose: boolean, visualises the variables in Tensorboard.
          ema_decay: scalar, the expontential decay rate for the moving average.
          steepness_factor: scalar, controls the routing mask weights.
          clip_threshold: scalar, threshold for clipping values in the mask.
    
        Returns:
          The routing mask and the routing ranks.
        """

        shp = R.shape
        if len(shp) == 5:
            capsule_routing = R.data.view(shp[0]*shp[1], shp[2], -1)
        else:
            capsule_routing = R.data.view(shp[0],shp[1],-1) #.view(shp[0], shp[1], -1)
        
        """ Calculate routing coefficients """
        # route: batch_size, input_dim, output_dim, dim_x, dim_y
        #capsule_routing = capsule_routing.sum(dim=-1) # batch-size*input_dim, output_dim

        """
        old=0
        bin=[]
        min = capsule_routing.min()
        iterations = 20
        step = (capsule_routing.max() - min) / iterations
        for i in range(iterations):
            val = (capsule_routing < (min + i*step)).sum().item() - old
            bin.append(val)
            old += val
        """
        
        
        """ Boosting """
        #if self.not_initialized:
        #    self.register_buffer('boosting_weights', torch.ones(capsule_routing.shape[1], device=R.device))
        #if self.training:
        #capsule_routing *= self.boosting_weights

        if self.lifetime:
            """ Rank routing coefficients """
            capsule_routing_sum = capsule_routing.sum(dim=-1) # batch-size*input_dim, output_dim
            order = capsule_routing_sum.sort(1, descending=True)[1]
            ranks = (-order).sort(1, descending=True)[1]
        
            #if self.training:
            """ Winning frequency """
            #transposed_ranks = ranks.transpose(1,0)  # output_dim, batch_size*input_dim
            if self.masked_freq:
                masked_routing = ((ranks < self.k).float() * capsule_routing_sum).sum(dim=0)
                freq = masked_routing / masked_routing.sum()
            else:
                win_counts = (ranks < self.k).sum(dim=0)
                freq = win_counts.float() / (self.k*ranks.shape[0]) # output_dim
        else:
            #order = capsule_routing.sort(1, descending=False)[1]
            #score = order.sort(1, descending=False)[1]
            #freq = score.float().mean(0) / capsule_routing.shape[1]
            
            numel = capsule_routing.shape[1]*capsule_routing.shape[2]
            mu = capsule_routing.sum(dim=(1,2))/numel
            sigma_sqr = ((capsule_routing-mu.view(-1,1,1)) ** 2).sum(dim=(1,2)) / numel
            activated_features = capsule_routing - capsule_routing * (capsule_routing < (mu+sigma_sqr.sqrt()).view(-1,1,1)).float()

            numel = (activated_features > 0).sum((1,2)).float()
            mu = activated_features.sum(dim=(1,2))/numel
            #sigma_sqr = ((activated_features-mu.unsqueeze(-1)) ** 2).sum(-1) / numel
            activated_features = activated_features - activated_features * (activated_features < mu.view(-1,1,1)).float()

            #activated_features = activated_features - ( activated_features * (activated_features < mask*activated_features      (activated_features.mean()+activated_features.std())).float() )
            activated_features = activated_features.sum(dim=-1)
            freq = (activated_features / activated_features.sum(dim=-1, keepdim=True).clamp(1e-10)).mean(dim=0)
    
        """ Moving average """
        if self.not_initialized:
            batch = capsule_routing_sum.shape[0]
            features = capsule_routing_sum.shape[1]
            self.freq_ema = freq
            self.target_max_freq /= features
            self.target_min_freq /= features
            #self.boost_update_count = int(self.boost_update_count / batch)
            self.ema_decay = self.ema_decay ** (1/self.boost_update_count)
            self.not_initialized = False

        self.freq_ema = self.ema_decay * self.freq_ema + (1 - self.ema_decay) * freq # output_dim

        #if (self.freq_ema != self.freq_ema).sum().item() > 0:
        #    print()

        self.N += 1

        if self.N == self.boost_update_count:
            self.N = 0

            if self.active:
                
                self.ramp_in = self.ramp_in-3/25 if self.ramp_in > 1 else self.ramp_in.new_tensor([1.])
                old = self.boosting_weights.clone()

                error_normalized = (self.freq_ema - self.target_max_freq) #/ self.target_max_freq
                error_normalized = (error_normalized > 0).float() * error_normalized
                correction = -error_normalized * self.down_boost_factor
                
                error_normalized = (-self.freq_ema + self.target_min_freq) #/ self.target_min_freq
                error_normalized = (error_normalized > 0).float() * error_normalized
                correction += error_normalized * self.up_boost_factor
                
                # DECAY
                error_normalized = (-self.freq_ema + self.target_max_freq)
                error_normalized = (error_normalized > 0).float() * error_normalized
                correction += error_normalized * self.up_boost_factor/10.
                nodecay = ((correction > 0) * (correction <= (error_normalized*self.up_boost_factor/10.))) != 1
                
                mean_compensation = 1. / ((self.boosting_weights < 1).float()*self.boosting_weights).mean().clamp(min=0.2)
                
                self.boosting_weights = self.boosting_weights + self.ramp_in * correction * self.boosting_weights * mean_compensation #.sqrt()
                self.boosting_weights = self.boosting_weights.clamp(min=0.05, max=1)

                
                """                     
                self.ramp_in = self.ramp_in-3/250 if self.ramp_in > 1 else self.ramp_in.new_tensor([1.])
                old = self.boosting_weights.clone()
                
                error_normalized = (self.freq_ema - self.target_max_freq) / self.target_max_freq  #).clamp(min=0, max=2) # 1 equal to 100% above
                self.down_error_integrated += error_normalized
                mask = (error_normalized > 0).float()
                self.down_error_integrated = (mask*self.down_error_integrated).clamp(max=1000.) # avoid wind-up
                error_normalized *= mask
                correction = -(error_normalized + self.down_error_integrated/3) * self.down_boost_factor
    
                error_normalized = (-self.freq_ema + self.target_min_freq) / self.target_min_freq  #).clamp(min=0, max=2) # 1 equal to 100% above
                self.up_error_integrated += error_normalized
                mask = (error_normalized > 0).float()
                self.up_error_integrated = (mask*self.up_error_integrated).clamp(max=1000.) # avoid wind-up
                error_normalized *= mask
                correction += (error_normalized + self.up_error_integrated/3) * self.up_boost_factor
                
                correction *= self.boosting_weights.clamp(min=0.01).sqrt()
                correction = (correction*100).int().float() / 100.
                
                self.boosting_weights = (self.boosting_weights + correction).clamp(min=0.01, max=1)
                """

            print ()
            print ()
            print ('freq_avg :', *[('%.2f' % i).lstrip('01').lstrip('.') for i in self.freq_ema.tolist()])
            if self.active:
                print ('boost    :', *[('%.2f' % i).lstrip('01').lstrip('.') for i in self.boosting_weights.tolist()])
                change = self.boosting_weights - old
                change = nodecay.float() * change
                change = (change==0).float() + change
                #s = str(['%.2f' % i for i in change.tolist()])
                #s = s.replace("["," ").replace("'","").replace("]","").replace(",","").replace(" 0","+").replace(" -0","-").replace(".00","    ") #.replace("1.00","    ")
                print ('change   :', *[('%.2f' % i).replace('-0.0','-').replace('0.0','+').replace('1.00','  ').replace('0',' ') for i in change.tolist()])
                #print ('Weights changed     :', (self.boosting_weights != old).sum().item())
                print ('Active filters (>5%) :', (self.freq_ema > 0.05/shp[1]).sum().item())


        #if not self.return_mask:
        #    return 1.

        #normalised_ranks = (ranks.shape[1]-1 - ranks).float() / (ranks.shape[1] - 1) # batch_size*input_dim, output_dim
        #routing_mask = torch.sigmoid(6.*(normalised_ranks - 0.5))
        #routing_mask = torch.exp(-self.steepness_factor * normalised_ranks) # batch_size*input_dim, output_dim
        #routing_mask = routing_mask - ( routing_mask * (routing_mask < self.clip_threshold).float() )
        #routing_mask = routing_mask.unsqueeze(-1).expand(shp)
        
        """
        updated_routing = mask.float() * capsule_routing
        mu = updated_routing.sum(dim=(1,2)) / mask.sum(dim=(1,2)).float()
        mu = (mu - 0.5) / 2. + 0.5
        return (updated_routing > mu.view(-1,1,1)).float()
        
        routing_mask = (ranks < self.sparsity*ranks.shape[1]).float()
        routing_mask = routing_mask.unsqueeze(-1).expand(shp) * activated_features_mask
        return routing_mask
        """
        
        
class MatrixRouting(nn.Module):
    def __init__(self, output_dim, num_routing, batchnorm=None, sparse=None, stat=None):
        super(MatrixRouting, self).__init__()
        self.output_dim = output_dim
        self.num_routing = num_routing
        self.beta_v = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1,1))
        self.sparse = sparse
        self.batchnorm = batchnorm
        self.sigmoid = nn.Sigmoid()
        #self.hardtanh = nn.Hardtanh(min_val=0,max_val=1)
        #if isinstance(activation, batchrenorm.Sigmoid):
        if sparse is None:
            self.beta_a = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1))
        #self.beta_aa = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1,1))
        #else:0
        #    self.beta_a = 0
        
        #self.experimental = experimental
        self.stat = stat
        if stat is not None:
            for _ in range(4):
                self.stat.append(0.)
        
    def forward(self, x): # (b, Bkk, Cww, h)
        #if type(votes) is tuple:
        """ the votes are pooled/reduced, so bias should be detached? """
        V = x[0]
        shp = V.shape

        """ If previous was ConvVector2d """
        if len(shp) > 5: # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
            vs = list(range(0, len(shp)))
            V = V.permute(vs[:3] + vs[4:] + [3]).contiguous()
            V = V.view(shp[0], shp[1], -1, shp[3])
            shp = V.shape

        b, Bkk, Cww, h = shp
        a_ = x[1].view(b, Bkk, -1)
        
        #V = v[...,:h-1]
        #a_ = v[...,h-1:].squeeze(-1)

        # routing coefficient
        if V.is_cuda:
            R = Variable(torch.ones(shp[:3]), requires_grad=False).cuda() / self.output_dim
        else:
            R = Variable(torch.ones(shp[:3]), requires_grad=False) / self.output_dim

        for i in range(self.num_routing):
            lambda_ = (1 - 0.65 ** (i+1)) * 1.37
            #lambda_ = 0.01 * (1 - 0.95 ** (i+1))
            #lambda_ = 1. * (1 - 0.5 ** (i+1))
            
            """ M-step: Compute an updated Gaussian model (μ, σ) """
            R = (R * a_).unsqueeze(-1)
            sum_R = R.sum(1) + eps
            mu = ((R * V).sum(1) / sum_R).unsqueeze(1)
            V_minus_mu_sqr = (V - mu) ** 2 + eps
            sigma_square = ((R * V_minus_mu_sqr).sum(1) / sum_R).unsqueeze(1) + eps

            """
            beta_v: Bias for log probability of sigma ("standard deviation")
            beta_a: Bias for offsetting
            
            In principle, beta_v and beta_a are only for learning regarding "activations".
            Just like "batch normalization" it has both scaling and bias.
            Votes are routed by the learned weight self.W.
            """
            log_sigma = torch.log(sigma_square.sqrt()+eps)

            if self.sparse is None:
                cost = (self.beta_v + log_sigma.view(b,self.output_dim,-1,h)) * sum_R.view(b, self.output_dim,-1,1)
                a = self.sigmoid(lambda_*(self.beta_a - cost.sum(-1)))
            else:
                is_last_time = (i == (self.num_routing - 1))
                #cost = (self.beta_v + self.beta_aa*log_sigma.view(b,self.output_dim,-1,h)) * sum_R.view(b, self.output_dim,-1,1)
                cost = (self.beta_v + log_sigma.view(b,self.output_dim,-1,h)) * sum_R.view(b, self.output_dim,-1,1)
                #a = self.batchnorm(self.beta_a - cost.sum(-1), i)
                a = self.batchnorm(-cost.sum(-1), i)
                #inp = lambda_*a*0.25 + 0.5
                #a = (inp.clamp(0.001,0.999) + inp*0.0001).clamp(0,1)
                
                a = self.sigmoid(lambda_*a)
                #a = (self.hardtanh(lambda_*a) + lambda_*a * 0.001 + 0.01).# *0.25+0.25)
                #a = ((lambda_*a).clamp(0.001, 0.999) + lambda_*a * 0.0001).clamp(0.,1.)
                #a = torch.sigmoid((lambda_*a - 0.5)/0.17)
                a = self.sparse.boosting_weights.view(1,-1,1) * a
                #a_boost = self.sparse.boosting_weights.view(1,-1,1) * a.data
                #a = a - (a_boost < 0.495).float() * a

                if is_last_time:
                    #mask = a.data > a.data.mean(dim=2, keepdim=True).clamp(min=0.495)
                    #a = mask.float() * a
                    #b = b * 0.00002 + 0.01
                    #a = torch.max(a, b)
                    
                    if self.training:
                        self.sparse.update(a) #sum_R.view(b, self.output_dim,-1)) * a
                        #if self.L_fac.item() < 0:
                        #    self.L_fac.data += 0.1

                        if self.sparse.N == 0:
                            bt99 = (a > 0.99).sum()
                            bt8 = (a > 0.8).sum() - bt99
                            bt5 = (a > 0.5).sum() - bt8 - bt99
                            bt2 = (a > 0.2).sum() - bt5 - bt8 - bt99
                            lt2 = (a > 0.01).sum() - bt2 - bt5 - bt8 - bt99
                            eq0 = (a == 0.).sum()
                            lt01 = (a < 0.01).sum() - eq0
                            print('Activity distribution:', eq0.item(), '|0|', lt01.item(), '|0.01|', lt2.item(), '|0.2|', bt2.item(), '|0.5|', bt5.item(), '|0.8|', bt8.item(), '|0.99|', bt99.item())
                            #if self.experimental==1:
                            #    print('Lfac =', self.L_fac.mean().item())
                    if self.stat is not None:
                        self.stat.append( log_sigma.std().item() )
                        cost_sum = cost.sum(-1)
                        self.stat.append( cost_sum.mean().item() )
                        self.stat.append( cost_sum.std().item() )
                        
                        mask_a = a>0.01
                        
                        numel_a = mask_a.sum(dim=2).float()+eps
                        mean_a = a.sum(dim=2)/numel_a
                        sigma_a = mask_a.float()*(a - mean_a.unsqueeze(-1)) ** 2
                        sigma_a = (sigma_a.sum(dim=2)/numel_a).sqrt()
                        sigma_a = sigma_a.mean()
                        
                        """
                        numel_a = mask_a.sum()
                        if numel_a.item() > 0:
                            mean_a = a.sum()/numel_a
                            sigma_a = mask_a.float() * (a - mean_a) ** 2
                            sigma_a = (sigma_a.sum()/numel_a).sqrt()
                        else:
                            sigma_a = numel_a.float()
                        """
                        
                        self.stat.append( sigma_a.item() )

            
            """ E-step: Recompute the assignment probabilities R(ij) based on the new Gaussian model and the new a(j) """
            if i != self.num_routing - 1:
                """ p_j_h is the probability of v_ij_h belonging to the capsule j’s Gaussian model """
                ln_p_j_h = -V_minus_mu_sqr / (2 * sigma_square) - log_sigma - 0.5*ln_2pi
                p_j_h = torch.exp(ln_p_j_h)

                """ Calculate normalized Assignment Probabilities (batch_size, input_dim, output_dim)"""
                ap = a.view(b, 1, Cww) * p_j_h.sum(-1)
                R = Variable(ap / (torch.sum(ap, 2, keepdim=True) + eps) + eps, requires_grad=False) # detaches from graph

        #mask = (a.data > 0).float()
        #mu = mask.view(b, 1, -1, 1) * mu
        return mu.view((b, self.output_dim) + x[0].shape[4:] + (-1,)), a.view((b, self.output_dim) + x[0].shape[4:] + (1,)), sum_R.view((b, self.output_dim) + x[0].shape[4:])
        #mu_a = torch.cat([mu.view((b, self.output_dim) + votes.shape[4:] + (-1,)), a.view((b, self.output_dim) + votes.shape[4:] + (1,))], dim=-1) # b, C, w, w, h


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, a_sz, r_sz, rnd_sz=0):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        self.a_sz = a_sz
        self.r_sz = r_sz
        self.rnd_sz = rnd_sz
        
    def forward(self, x):
        """
        x:     batch_size, output_dim, dim_x, dim_y, h
        route: batch_size, input_dim, output_dim, dim_x, dim_y
        votes: batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
        """
        x, route, votes = x
        b, input_dim, output_dim, h, dim_x, dim_y = votes.shape
        a_orig = x[:,:,:,:,h-1]   # b, output_dim, w_x, w_y

        a, a_sort_id = self.maxpool(a_orig)
        a_sort_id = a_sort_id.view(b, output_dim, -1)
        sort_id = a.view(b, output_dim, -1).sort(2, descending=True)[1] # batch, output_dim, dim_x*dim_y
        a_sort_id = a_sort_id.gather(2, sort_id[:,:,:self.a_sz])
        
        offset = torch.arange(b*output_dim, device=x.device).unsqueeze(-1).repeat(1,a_sort_id.shape[-1]) * dim_x*dim_y
        offset = offset.view_as(a_sort_id)
        mask = torch.zeros(a_orig.shape, device=x.device)
        a_sort_id_trans = a_sort_id + offset
        mask.view(-1)[a_sort_id_trans.view(-1)] = 1

        if self.rnd_sz != 0:
            idx_lucky = torch.randperm(b*output_dim*dim_x*dim_y)[:self.rnd_sz]
            mask.view(-1)[idx_lucky] = 1
        
        mask = mask.unsqueeze(1).repeat(1,input_dim,1,1,1)
        route = route + mask

        route = route.view(b, -1, dim_x, dim_y)
        route, r_sort_id = self.maxpool(route)
        r_sort_id = r_sort_id.view(b, input_dim, output_dim, -1)
        sort_id = route.view(b, input_dim, output_dim, -1).sort(3, descending=True)[1] # batch, output_dim, dim_x*dim_y
        r_sort_id = r_sort_id.gather(3, sort_id[:,:,:,:self.a_sz+self.r_sz+self.rnd_sz])

        idx = torch.randperm(r_sort_id.shape[-1])
        r_sort_id = r_sort_id[:,:,:,idx]

        ids_combined_h = r_sort_id[:,:,:,None,:].repeat(1,1,1,h,1) # batch, input_dim, output_dim, h, dim_x*dim_y

        votes = votes.view(b,input_dim,output_dim,h,-1).gather(4, ids_combined_h).unsqueeze(-1)

        return votes, None # is None, to force bias detach


class MaxActPool(nn.Module):
    def __init__(self, kernel_size, stride=None, out_sz=100, output_ids=True):
        super(MaxActPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        self.out_sz = out_sz
        self.output_ids = output_ids
        
    def forward(self, x):
        b, output_dim, dim_x, dim_y, h = x.shape
        a = x[:,:,:,:,h-1]   # b, output_dim, w_x, w_y

        a, sort_id_mp = self.maxpool(a)
        sort_id_h = sort_id_mp.view(b, output_dim, -1)[:,:,:,None].repeat(1,1,1,h)
        x = x.view(b, output_dim, -1, h).gather(2, sort_id_h)

        sort_id = a.view(b, output_dim, -1).sort(2, descending=True)[1] # batch, output_dim, dim_x*dim_y
        sort_id_h = sort_id[:,:,:,None].repeat(1,1,1,h) # batch, output_dim, dim_x*dim_y, h
        x = x.gather(2, sort_id_h[:,:,:self.out_sz,:]).unsqueeze(3)
        if self.output_ids:
            sorted_ids = sort_id_mp.view(b,output_dim,-1).gather(2, sort_id)[:,:,:self.out_sz]
            return x, sorted_ids, dim_x, dim_y
        return x


class KeyRouting(nn.Module):
    def __init__(self, conv_layer, pre_conv_x_obj):
        super(KeyRouting, self).__init__()
        self.conv_layer = conv_layer
        self.pre_conv_x_obj = pre_conv_x_obj

    def forward(self, x): # x: # # batch_size*input_dim, input_atoms, dim_x, dim_y
        # Create Keys and center these with a mean of zero
        shp = self.conv_layer.conv.weight.size()
        mean_weight = self.conv_layer.conv.weight.data.view(shp[0],shp[1],-1).mean(2, keepdim=True).unsqueeze(-1).expand(shp)
        corr_weight = (self.conv_layer.conv.weight.data - mean_weight)
        del mean_weight
        
        # Normalize keys, so high contrast keys don't contribute more than low contrast keys
        norm = corr_weight.view(shp[0],shp[1],-1).norm(p=1, dim=2, keepdim=True).unsqueeze(-1)
        norm_weight = corr_weight / norm
        del corr_weight
        del norm
        
        # Center x with a mean of zero
        x_shp = self.pre_conv_x_obj[0].size()
        pre_x = self.pre_conv_x_obj[0].data.view(x_shp[0]*x_shp[1], x_shp[2], x_shp[3], x_shp[4])  # batch_size*input_dim, input_atoms, dim_x, dim_y
        mean_x = pre_x.view(x_shp[0]*x_shp[1],x_shp[2],-1).mean(dim=2, keepdim=True).unsqueeze(-1)
        corr_x = pre_x - mean_x
        del mean_x

        # Convolve coor_x - Try to fit keys
        corr_x = F.conv2d(corr_x, norm_weight, None, self.conv_layer.conv.stride, self.conv_layer.conv.padding, self.conv_layer.conv.dilation, 1) # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        del norm_weight
        
        a_sort = corr_x.view(x_shp[0]*x_shp[1], x.shape[2], x.shape[3], -1).norm(p=1, dim=2).norm(p=2, dim=1) #.sort(1, descending=True)[1]
        #order = a_sort.sort(1)[1]
        #rank = (-order).sort(1,descending=True)[1].float()
        # batch_size*input_dim, output_dim, h, out_dim_x*out_dim_y -> batch_size*input_dim, out_dim_x*out_dim_y

        #del corr_x
        a_sort = a_sort.view(x_shp[0], x_shp[1], 1, corr_x.shape[-2], corr_x.shape[-1]).repeat(1,1,x.shape[2],1,1) # batch_size, input_dim, output_dim, out_dim_x, out_dim_y

        return x, a_sort


def dynamic_routing(votes, biases, logit_shape, num_routing):
    # votes: batch_size, input_dim, output_dim, h, (dim_x, dim_y)
    
    votes_trans = votes.permute(votes_t_shape)                      # h, batch_size, input_dim, output_dim, (dim_x, dim_y)
    #votes_trans_stopped = votes_trans.clone().detach()

    logits = Variable(torch.zeros(logit_shape, device=votes.device), requires_grad=False)
                                                                    # batch_size, input_dim, output_dim, (dim_x, dim_y)
    for i in range(num_routing):
        route = F.softmax(logits, 2)                                # batch_size, input_dim, output_dim, (dim_x, dim_y)

        if i == num_routing - 1:
            preactivate_unrolled = route * votes_trans              # h, batch_size, input_dim, output_dim, (dim_x, dim_y)
            preact_trans = preactivate_unrolled.permute(r_t_shape)  # batch_size, input_dim, output_dim, h, (dim_x, dim_y)
            preactivate = torch.sum(preact_trans, dim=1) + biases   # batch_size, output_dim, h, (dim_x, dim_y)
            activation = _squash(preactivate)                       # squashing of "h" dimension
        else:
            preactivate_unrolled = route * votes_trans.data #_stopped      # h, batch_size, input_dim, output_dim, (dim_x, dim_y)
            preact_trans = preactivate_unrolled.permute(r_t_shape)  # batch_size, input_dim, output_dim, h, (dim_x, dim_y)
            preactivate = torch.sum(preact_trans, dim=1) + biases   # batch_size, output_dim, h, (dim_x, dim_y)
            activation = _squash(preactivate)                       # squashing of "h" dimension
            act_3d = activation.unsqueeze_(1)                       # batch_size, 1, output_dim, h, (dim_x, dim_y)
            distances = torch.sum(votes.data * act_3d, dim=3)            # batch_size, input_dim, output_dim, (dim_x, dim_y)
            logits = logits + distances                             # batch_size, input_dim, output_dim, (dim_x, dim_y)
            
    return (activation, route)                                        # batch_size, output_dim, h, (dim_x, dim_y)

def _squash(input_tensor):
    norm = torch.norm(input_tensor, p=2, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
