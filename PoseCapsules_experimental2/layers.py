import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from batchrenorm import BatchRenorm
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
        
    #return int( (s*(i-1) - i + k + (k-1)*(d-1)) / 2 )
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
        x = x[0]
        return x.permute(0,1,4,2,3).contiguous().view(x.shape[0], -1, x.shape[2], x.shape[3])

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


class PrimMatrix2d(nn.Module):
    def __init__(self, output_dim, h, kernel_size, stride, padding, bias, advanced=False, func='Conv2d'):
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
        
    def init(self, x): # batch_size, input_dim, input_atoms, dim_x, dim_y
        in_channels = 1 #int(x.size(2)/(self.h+1))
        in_h = x.shape[2] - 1
        ConvFunc = getattr(nn,self.func)
        
        if self.kernel_size == 0:
            self.kernel_size = x.shape[-1]

        if self.advanced:
            #y = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])  # batch_size*input_dim, input_atoms, dim_x, dim_y
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
            x = x[0]
            shp = x.shape                                           # batch_size, input_dim, input_atoms, dim_x, dim_y
            if shp[-3] == shp[-2] and shp[-2] != shp[-1]:
                """ If previous was MatrixRouting """
                s = list(range(0, len(shp)-1))
                x = x.permute(s[:2] + [-1] + s[2:])
        else:
            """ Previous was Conv2d """
            x = x.unsqueeze(1)
        shp = x.shape
            
        if self.not_initialized:
            self.init(x)
        in_h = shp[2] - 1
        
        x = x.view((shp[0]*shp[1],) + shp[2:])  # batch_size*input_dim, input_atoms, dim_x, dim_y

        if self.advanced:
            votes = self.pad_pre(x[:,:in_h,...])
            votes = self.conv_pre(votes)
            votes = F.relu(votes, inplace=True)
            votes = torch.cat([x[:,:in_h,...], votes], dim=1)
            votes = self.conv(votes)                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        else:
            votes = self.conv(x[:,:in_h,...])                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        votes = votes.view((shp[0], -1, self.output_dim, self.h) + votes.shape[2:])

        activations = self.conv_a(x[:,in_h:,...])                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        #activations = torch.sigmoid(activations)
        activations = activations.view((shp[0], -1, 1, 1) + activations.shape[2:]).expand(votes.shape[:3] + (1,) + votes.shape[4:])

        x = torch.cat([votes, activations], dim=3) # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
        return x


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
        #x = x.permute(0,1,3,4,2) # move hh to the end -> batch, input_dim, dim_x, dim_y, input_atoms
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


class MatrixRouting(nn.Module):
    def __init__(self, output_dim, num_routing):
        super(MatrixRouting, self).__init__()
        self.output_dim = output_dim
        self.num_routing = num_routing
        self.beta_v = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1,1))
        self.beta_a = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1))
        
    def forward(self, votes): # (b, Bkk, Cww, h)
        if type(votes) is tuple:
            """ the votes are pooled/reduced, so bias should be detached? """
            votes = votes[0]
        shp = votes.shape

        """ If previous was ConvVector2d """
        if len(shp) > 5: # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
            vs = list(range(0, len(shp)))
            v = votes.permute(vs[:3] + vs[4:] + [3]).contiguous()
            v = v.view(shp[0], shp[1], -1, shp[3])
            w = shp[-2]
            shp = v.shape
        else:
            w = int(math.sqrt(shp[2] / self.output_dim))
            v = votes

        b, Bkk, Cww, h = shp
        
        V = v[...,:h-1]
        a_ = v[...,h-1:].squeeze(-1)

        # routing coefficient
        if V.is_cuda:
            R = Variable(torch.ones(shp[:3]), requires_grad=False).cuda() / self.output_dim
        else:
            R = Variable(torch.ones(shp[:3]), requires_grad=False) / self.output_dim

        for i in range(self.num_routing):
            lambda_ = 0.01 * (1 - 0.95 ** (i+1))
            
            """ M-step: Compute an updated Gaussian model (μ, σ) """
            R = (R * a_).unsqueeze(-1)
            sum_R = R.sum(1) + 0.0001
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
            cost = (self.beta_v + log_sigma.view(b,self.output_dim,-1,h-1)) * sum_R.view(b, self.output_dim,-1,1)
            a = torch.sigmoid(lambda_ * (self.beta_a - cost.sum(-1)))
            a = a.view(b, Cww)

            """ E-step: Recompute the assignment probabilities R(ij) based on the new Gaussian model and the new a(j) """
            if i != self.num_routing - 1:
                """ p_j_h is the probability of v_ij_h belonging to the capsule j’s Gaussian model """
                ln_p_j_h = -V_minus_mu_sqr / (2 * sigma_square) - log_sigma - 0.5*ln_2pi
                p_j_h = torch.exp(ln_p_j_h)

                """ Calculate normalized Assignment Probabilities (batch_size, input_dim, output_dim)"""
                ap = a[:,None,:] * p_j_h.sum(-1)
                R = Variable(ap / (torch.sum(ap, 2, keepdim=True) + eps) + eps, requires_grad=False) # detaches from graph

        mu_a = torch.cat([mu.view((b, self.output_dim) + votes.shape[4:] + (-1,)), a.view((b, self.output_dim) + votes.shape[4:] + (1,))], dim=-1) # b, C, w, w, h

        return mu_a, None #R.view((b, Bkk, self.output_dim) + votes.shape[4:]), votes


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

        #ids_combined = torch.cat([a_sort_id[:,None,:,:].repeat(1,input_dim,1,1), r_sort_id], dim=3)
        ids_combined_h = r_sort_id[:,:,:,None,:].repeat(1,1,1,h,1) # batch, input_dim, output_dim, h, dim_x*dim_y

        votes = votes.view(b,input_dim,output_dim,h,-1).gather(4, ids_combined_h).unsqueeze(-1)
        #idx = torch.randperm(votes.shape[4])
        #votes = votes[:,:,:,:,idx].unsqueeze(-1)        # batch, input_dim, output_dim, h, dim_x, dim_y

        return votes, None # is None, to force bias detach


class MaxActPool(nn.Module):
    def __init__(self, kernel_size, stride=None, out_sz=100, output_ids=True):
        super(MaxActPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        self.out_sz = out_sz
        self.output_ids = output_ids
        
    def forward(self, x):
        #x, _, _ = x
        b, output_dim, dim_x, dim_y, h = x.shape
        #x = x.view(b, output_dim, -1, h)
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
