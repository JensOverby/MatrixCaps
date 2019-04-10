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
lambda_ = 0.0001

def calc_out(input_, kernel=1, stride=1, padding=0, dilation=1):
    return int((input_ + 2*padding - dilation*(kernel-1) - 1) / stride) + 1

def calc_same_padding(i, k=1, s=1, d=1, transposed=False):
    return int( (s*(i-1) - i + k + (k-1)*(d-1)) / 2 )
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
        return x[0][...,:self.output_dim], None, None


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

            """
            sz = 1
            while sz < w:
                sz *= 2
            row = torch.zeros(sz)
            sz0 = sz

            row = torch.zeros(sz)
            while sz > 8:
                for i in range(sz0):
                    theta = 2*math.pi*i/sz
                    row[i] += math.sin(theta)
                sz /= 2

            for i in range(sz0):
                print(row[i].item())

            """
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
        elif (shp[-1] == self.h) and (shp[2] == shp[3]): # b, C, w, w, hh
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
    def __init__(self, output_dim, h, kernel_size, stride, padding, bias, activate=False):
        super(PrimMatrix2d, self).__init__()
        self.output_dim = output_dim
        self.h = h
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activate = activate
        self.not_initialized = True
        
    def init(self, x): # batch_size, input_dim, input_atoms, dim_x, dim_y
        in_channels = int(x.size(2)/(self.h+1))
        self.conv = nn.Conv2d(in_channels=in_channels*self.h,
                                       out_channels=self.output_dim * self.h,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.padding,
                                       bias=self.bias)
        nn.init.normal_(self.conv.weight.data, mean=0,std=0.1)
        if self.bias:
            nn.init.normal_(self.conv.bias.data, mean=0,std=0.1)
        if self.activate:
            self.conv_a = nn.Conv2d(in_channels=in_channels,
                                           out_channels=self.output_dim,
                                           kernel_size=self.kernel_size,
                                           stride=self.stride,
                                           padding=self.padding,
                                           bias=self.bias)
            nn.init.normal_(self.conv_a.weight.data, mean=0,std=0.1)
            if self.bias:
                nn.init.normal_(self.conv_a.bias.data, mean=0,std=0.1)
        else:
            self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=self.padding)
        
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
        elif (shp[-1] == (self.h+1)) and (shp[2] == shp[3]): # b, C, w, w, hh
            """ If previous was MatrixRouting """
            x = x.permute(0,1,4,2,3)
            shp = x.shape
        elif len(shp) == 6:
            shp = list(shp)
            shp[1] = shp[1]*shp[2]
            shp.pop(2)
            x = x.view(shp)
        if self.not_initialized:
            self.init(x)
        
        x = x.view(shp[0]*shp[1], shp[2], shp[3], shp[4])  # batch_size*input_dim, input_atoms, dim_x, dim_y

        votes = self.conv(x[:,:self.h,:,:])                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
        #if not self.activate:
        #    votes = torch.tanh(votes)
        votes = votes.view(shp[0], -1, self.output_dim, self.h, votes.size(-2), votes.size(-1))

        if self.activate:
            activations = self.conv_a(x[:,self.h:,:,:])                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y
            activations = torch.sigmoid(activations)
            activations = activations.view(shp[0], -1, self.output_dim, 1, votes.size(-2), votes.size(-1))
        else:
            activations = self.pool(x[:,self.h:,:,:])
            activations = activations.view(shp[0], -1, 1, 1, activations.size(-2), activations.size(-1)).repeat(1,1,self.output_dim,1,1,1)
            #shp = list(votes.shape)
            #shp.pop(3)
            #activations = torch.ones(shp, device=votes.device).unsqueeze(3)

        x = torch.cat([votes, activations], dim=3)
        #x = x.view(shp[0], -1, self.output_dim, self.h, x.size(-2), x.size(-1)) # batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y

        return x


class ConvMatrix2d(nn.Module):
    def __init__(self, output_dim, hh, kernel_size, stride, padding):
        super(ConvMatrix2d, self).__init__()
        self.output_dim = output_dim
        self.h = int(math.sqrt(hh))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.not_initialized = True

    def init(self, x): # batch, input_dim, input_atoms, dim_x, dim_y       (self.b, self.C, self.w, self.w, hh)
        self.weight = nn.Parameter(torch.randn(x.shape[1], self.kernel_size, self.kernel_size, self.output_dim, self.h, self.h))  # B,K,K,C,4,4
        self.not_initialized = False

    def forward(self, x): # (self.b, self.C, self.w, self.w, hh)    # batch, input_dim, input_atoms, dim_x, dim_y
        shp = x.shape
        
        """ If previous was ConvVector2d """
        if len(shp) == 6: # batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
            x = x.permute(0,1,2,4,5,3)
            x = x.view(shp[0], -1, shp[-2], shp[-1], shp[3])
        if self.not_initialized:
            self.init(x)

        b, input_dim, w, _, hh = x.shape
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
            
            """ HACK """
            #kaj = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
            #idx = torch.randperm(kaj.shape[-1])
            #x = kaj[:,:,:,idx].view_as(x)        # batch, input_dim, output_dim, h, dim_x, dim_y
            
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

    def init(self, x): # b, B, hh
        self.weight = nn.Parameter(torch.randn(x.shape[1], self.C, self.h, self.h))
        self.not_initialized = False

    def forward(self, x): # b, B, hh
        if type(x) is tuple:
            x = x[0]
        shp = x.shape
        if len(shp) == 5:
            if (shp[-2] == shp[-1]) or (shp[-1] == 1):
                x = x.view(shp[0], shp[1], -1)
            else:
                x = x.view(shp[0], -1, shp[-1])
        if self.not_initialized:
            self.init(x)

        b, B, hh = x.shape

        activations = x[...,hh-1:]
        poses = x[...,:hh-1].view(b, B, 1, self.h, self.h)
        
        votes = self.weight @ poses

        votes = votes.view(b, B, self.C, -1)
        activations = activations.view(b, B, 1, 1).repeat(1, 1, self.C, 1)
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
        #self.not_initialized = True

    #def init(self, votes):
    #    self.not_initialized = False
        
    def forward(self, votes): # (b, Bkk, Cww, hh)
        if type(votes) is tuple:
            """ the votes are pooled/reduced, so bias should be detached? """
            votes = votes[0]
        shp = votes.shape

        """ If previous was ConvVector2d """
        if len(shp) == 6: # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
            v = votes.permute(0,1,2,4,5,3).contiguous()
            v = v.view(shp[0], shp[1], shp[2]*shp[4]*shp[5], shp[3])
            w_x, w_y = shp[-2], shp[-1]
            #if shp[-2] != shp[-1]:
            #    self.output_dim = v.shape[2]
            shp = v.shape
        else:
            w_x = w_y = int(math.sqrt(shp[2] / self.output_dim))
            v = votes

        #if self.not_initialized:
        #    self.init(v)
        b, Bkk, Cww, hh = shp
        
        V = v[...,:hh-1]
        a_ = v[...,hh-1:].squeeze(-1)

        # routing coefficient
        if V.is_cuda:
            R = Variable(torch.ones(shp[:3]), requires_grad=False).cuda() / self.output_dim
        else:
            R = Variable(torch.ones(shp[:3]), requires_grad=False) / self.output_dim

        for i in range(self.num_routing):
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
            cost = (self.beta_v + log_sigma.view(b,self.output_dim,-1,hh-1)) * sum_R.view(b, self.output_dim,-1,1)
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

        mu_a = torch.cat([mu.view(b, self.output_dim, w_x, w_y, -1), a.view(b, self.output_dim, w_x, w_y, 1)], dim=-1) # b, C, w, w, hh

        return mu_a, R.view(b, Bkk, self.output_dim, w_x, w_y), votes


class MaxRoutePool(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxRoutePool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        
    def forward(self, x):
        """
        votes: batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
        route: batch_size, input_dim, output_dim, (dim_x, dim_y)
        """

        """ If previous was VectorRouting """
        if len(x) == 3:
            y, route, votes = x
        else:
            votes, route = x
        
        #routing = x[1].max(dim=1)[0] # batch, output_dim, dim_x, dim_y
        #routing = x[1].sum(dim=1) # batch, output_dim, dim_x, dim_y
        route = route.view(-1, route.shape[2], route.shape[3], route.shape[4])
        _, indices = self.maxpool(route)
        v_shp = votes.shape
        i_shp = indices.shape
        indices = indices.view(i_shp[0],i_shp[1],-1)
        route = route.view(i_shp[0],i_shp[1],-1).gather(2, indices).view(v_shp[0],v_shp[1],i_shp[1],i_shp[2],i_shp[3])
        indices = indices.view(v_shp[0], v_shp[1], i_shp[1], 1, -1).repeat(1,1,1,v_shp[3],1)
        votes = votes.view(v_shp[0],v_shp[1],v_shp[2],v_shp[3],-1).gather(4, indices).view(v_shp[0],v_shp[1],v_shp[2],v_shp[3],i_shp[-2],i_shp[-1])
        return (votes, route)

class MaxRouteReduce(nn.Module):
    def __init__(self, out_sz, max_pct=37., sum_pct=37.):
        super(MaxRouteReduce, self).__init__()
        self.max_sz = int(max_pct*out_sz/100.)
        self.sum_sz = int(sum_pct*out_sz/100.)
        self.rnd_sz = out_sz - self.max_sz - self.sum_sz
        
    def forward(self, x):
        """ Calculate routing coefficients """
        # votes: batch, input_dim, output_dim, h, dim_x, dim_y
        # route: batch, input_dim, output_dim, dim_x, dim_y

        """ If previous was VectorRouting """
        if len(x) == 3:
            _, route, votes = x
        else:
            votes, route = x

        b, input_dim, output_dim, h, _, _ = votes.shape
        votes = votes.view(b, input_dim, output_dim, h, -1)
        route = route.view(b, input_dim, output_dim, -1) # batch, input_dim, output_dim, dim_x*dim_y

        segment_list = []

        """
        if self.a_sum_sz != 0:
            a = x[:,:,:,:,h-1].view(b, output_dim, -1) # b, output_dim, w_x*w_y
            oper = a.sum(dim=1) # batch, dim_x*dim_y
            sort_id = oper.sort(1, descending=True)[1] 
            sort_id = sort_id[:,None,None,:].repeat(1,input_dim,output_dim,1)# batch, input_dim, output_dim, dim_x*dim_y
            sort_id_h = sort_id[:,:,:,None,:].repeat(1,1,1,h,1)# batch, input_dim, output_dim, h, dim_x*dim_y
            segment_list.append( votes.gather(4, sort_id_h[:,:,:,:,:self.a_sum_sz]) )
            votes = votes.gather(4, sort_id_h[:,:,:,:,self.a_sum_sz:])
            route = route.gather(3, sort_id[:,:,:,self.a_sum_sz:])
        """

        if self.max_sz != 0:
            oper = route.max(dim=2)[0] # batch, input_dim, dim_x*dim_y
            sort_id = oper.sort(2, descending=True)[1] # batch, input_dim, dim_x*dim_y
            sort_id = sort_id[:,:,None,:].repeat(1,1,output_dim,1) # batch, input_dim, output_dim, dim_x*dim_y
            sort_id_h = sort_id[:,:,:,None,:].repeat(1,1,1,h,1) # batch, input_dim, output_dim, h, dim_x*dim_y
            segment_list.append( votes.gather(4, sort_id_h[:,:,:,:,:self.max_sz]) )
            votes = votes.gather(4, sort_id_h[:,:,:,:,self.max_sz:])
            route = route.gather(3, sort_id[:,:,:,self.max_sz:])

        if self.sum_sz != 0:
            oper = route.sum(dim=2)
            sort_id = oper.sort(2, descending=True)[1] # batch, input_dim, dim_x*dim_y
            sort_id = sort_id[:,:,None,:].repeat(1,1,output_dim,1) # batch, input_dim, output_dim, dim_x*dim_y
            sort_id_h = sort_id[:,:,:,None,:].repeat(1,1,1,h,1) # batch, input_dim, output_dim, h, dim_x*dim_y
            segment_list.append( votes.gather(4, sort_id_h[:,:,:,:,:self.sum_sz]) )
            votes = votes.gather(4, sort_id_h[:,:,:,:,self.sum_sz:])
            #route = route.gather(3, sort_id[:,:,:,self.sum_sz:])

        if self.rnd_sz != 0:
            idx_lucky = torch.randperm(votes.shape[4])[:self.rnd_sz]
            segment_list.append( votes[:,:,:,:,idx_lucky] )

        votes = torch.cat(segment_list, dim=4)

        idx = torch.randperm(votes.shape[4])
        votes_sorted = votes[:,:,:,:,idx].unsqueeze(-1)        # batch, input_dim, output_dim, h, dim_x, dim_y

        return votes_sorted, None # is None, to force bias detach



        """
        if len(x) == 3:
            votes, route = x[2], x[1]
        else:
            votes, route = x[0], x[1]
        
        v_shp = votes.shape
        
        route_max = route.max(dim=2)[0]
        route_sort_id = route_max.view(v_shp[0], v_shp[1], -1).sort(2, descending=True)[1] # batch, input_dim, dim_x*dim_y
        route_sort_best_id = route_sort_id[:,:,None,None,:self.max_sz].repeat(1,1,v_shp[2],v_shp[3],1)
        x_best = votes.view(v_shp[0], v_shp[1], v_shp[2], v_shp[3], -1).gather(4, route_sort_best_id)
        
        route_sort_rest_id = route_sort_id[:,:,None,self.max_sz:].repeat(1,1,v_shp[2],1)
        route_rest = route.view(v_shp[0],v_shp[1],v_shp[2],-1).gather(3,route_sort_rest_id)
        
        #capsule_routing = x[1].view(-1, x_sh[1], x_sh[-2], x_sh[-1])
        route_sum = route_rest.sum(dim=2) # batch, input_dim, dim_x, dim_y

        route_sort_id = route_sum.view(v_shp[0], v_shp[1], -1).sort(2, descending=True)[1] # batch, input_dim, dim_x*dim_y
        route_sort_id = route_sort_id[:,:,None,None,:].repeat(1,1,v_shp[2],v_shp[3],1)

        x_sorted = votes.view(v_shp[0], v_shp[1], v_shp[2], v_shp[3], -1).gather(4, route_sort_id)
        best = x_sorted[:,:,:,:,:self.sum_sz]
        rest = x_sorted[:,:,:,:,self.sum_sz:]
        idx_lucky = torch.randperm(rest.size(4))[:self.rnd_sz]
        x_sorted = torch.cat([x_best,best,rest[:,:,:,:,idx_lucky]], dim=4)

        idx = torch.randperm(self.max_sz+self.sum_sz+self.rnd_sz)
        x_sorted = x_sorted[:,:,:,:,idx].unsqueeze(-1)        # batch, input_dim, output_dim, h, dim_x, dim_y

        return x_sorted, None # is None, to force bias detach
        """

"""
class MaxRouteReduce(nn.Module):
    def __init__(self, route_max_sz=37, route_sum_sz=38, a_max_sz=37, a_sum_sz=38, rnd_sz=0, kernel_size=0, stride=0):
        super(MaxRouteReduce, self).__init__()
        self.route_max_sz=route_max_sz
        self.route_sum_sz=route_sum_sz
        self.a_max_sz=a_max_sz
        self.a_sum_sz=a_sum_sz
        self.rnd_sz=rnd_sz
        
        if kernel_size!=0 and stride!=0:
            self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        else:
            self.maxpool = None

    
    def forward2(self, x):
        #if self.use_old:
        #    return self.forward2(x)

        shp = x[0].shape
        a = x[0][:,:,:,:,shp[-1]-1:]   # b, output_dim, w_x, w_y, 1
        a_max = a.max(dim=1)[0] # b, self.max_sz, 1
        a_sort_id = a_max.view(shp[0], -1).sort(1, descending=True)[1]
        a_sort_id = a_sort_id[:,None,:,None].repeat(1,shp[1],1,shp[-1])
        x_best = x[0].view(shp[0], shp[1], -1, shp[-1]).gather(2, a_sort_id[:,:,:self.max_sz,:])

        x_rest = x[0].view(shp[0], shp[1], -1, shp[-1]).gather(2, a_sort_id[:,:,self.max_sz:,:])

        a = x_rest[:,:,:,shp[-1]-1:]   # b, output_dim, self.max_sz, 1
        a_sum = a.sum(dim=1) # b, self.max_sz, 1
        a_sort_id = a_sum.view(shp[0], -1).sort(1, descending=True)[1]
        a_sort_id = a_sort_id[:,None,:,None].repeat(1,shp[1],1,shp[-1])
        x_2best = x_rest.gather(2, a_sort_id[:,:,:self.sum_sz,:])

        x_rest = x_rest.gather(2, a_sort_id[:,:,self.sum_sz:,:])

        idx_lucky = torch.randperm(x_rest.size(2))[:self.rnd_sz]
        x_sorted = torch.cat([x_best,x_2best,x_rest[:,:,idx_lucky,:]], dim=2)

        idx = torch.randperm(self.max_sz+self.sum_sz+self.rnd_sz)
        x_sorted = x_sorted[:,:,idx,:].unsqueeze(3)        # batch, output_dim, dim_x*dim_y, 1, h
        return x_sorted

    def forward(self, x):
        x, route, _ = x
        b, output_dim, _, _, h = x.shape
        x = x.view(b, output_dim, -1, h)
        a = x[:,:,:,h-1]   # b, output_dim, w_x, w_y
        route = route.sum(1)

        segment_list = []
        
        if self.maxpool is not None:
            _, sort_id = self.maxpool(route)
            sort_id = sort_id.view(b, output_dim, -1)
            route = route.view(b, output_dim, -1)
            sort_id_h = sort_id[:,:,:,None].repeat(1,1,1,h)
            segment_list.append( x.gather(2, sort_id_h) )

            offset = torch.arange(b*output_dim, device=x.device).unsqueeze(-1).repeat(1,sort_id.shape[-1]) * x.shape[-2]
            offset = offset.view_as(sort_id)
            mask = torch.ones(route.shape, device=x.device, dtype=torch.uint8)
            sort_id_trans = sort_id + offset
            mask.view(-1)[sort_id_trans.view(-1)] = 0
            route = torch.masked_select(route, mask).view(b, output_dim, -1)
            a = torch.masked_select(a, mask).view(b, output_dim, -1)
            mask = mask[:,:,:,None].repeat(1,1,1,h)
            x = torch.masked_select(x, mask).view(b, output_dim, -1, h)
        else:
            route = route.view(b, output_dim, -1)
            
        if self.route_max_sz != 0:
            oper = route.max(dim=1)[0]
            sort_id = oper.sort(1, descending=True)[1] # batch, input_dim, dim_x*dim_y
            sort_id = sort_id[:,None,:].repeat(1,output_dim,1)
            sort_id_h = sort_id[:,:,:,None].repeat(1,1,1,h)
            segment_list.append( x.gather(2, sort_id_h[:,:,:self.route_max_sz,:]) )
            x = x.gather(2, sort_id_h[:,:,self.route_max_sz:,:])
            route = route.gather(2,sort_id[:,:,self.route_max_sz:])
            a = a.gather(2,sort_id[:,:,self.route_max_sz:])

        if self.route_sum_sz != 0:
            oper = route.sum(dim=1)
            sort_id = oper.sort(1, descending=True)[1] # batch, input_dim, dim_x*dim_y
            sort_id = sort_id[:,None,:].repeat(1,output_dim,1)
            sort_id_h = sort_id[:,:,:,None].repeat(1,1,1,h)
            segment_list.append( x.gather(2, sort_id_h[:,:,:self.route_sum_sz,:]) )
            x = x.gather(2, sort_id_h[:,:,self.route_sum_sz:,:])
            #route = route.gather(2,sort_id[:,:,self.route_sum_sz:])
            a = a.gather(2,sort_id[:,:,self.route_sum_sz:])


        if self.a_max_sz != 0:
            oper = a.max(dim=1)[0]
            sort_id = oper.sort(1, descending=True)[1] # batch, input_dim, dim_x*dim_y
            sort_id = sort_id[:,None,:].repeat(1,output_dim,1)
            sort_id_h = sort_id[:,:,:,None].repeat(1,1,1,h)
            segment_list.append( x.gather(2, sort_id_h[:,:,:self.a_max_sz,:]) )
            x = x.gather(2, sort_id_h[:,:,self.a_max_sz:,:])
            #route = route.gather(2,sort_id[:,:,self.a_max_sz:])
            a = a.gather(2,sort_id[:,:,self.a_max_sz:])

        if self.a_sum_sz != 0:
            oper = a.sum(dim=1)
            sort_id = oper.sort(1, descending=True)[1] # batch, input_dim, dim_x*dim_y
            sort_id = sort_id[:,None,:].repeat(1,output_dim,1)
            sort_id_h = sort_id[:,:,:,None].repeat(1,1,1,h)
            segment_list.append( x.gather(2, sort_id_h[:,:,:self.a_sum_sz,:]) )
            x = x.gather(2, sort_id_h[:,:,self.a_sum_sz:,:])
            #route = route.gather(2,sort_id[:,:,self.a_sum_sz:])
            #a = a.gather(2,sort_id[:,:,self.a_sum_sz:])

        if self.rnd_sz != 0:
            idx_lucky = torch.randperm(x.shape[2])[:self.rnd_sz]
            segment_list.append( x[:,:,idx_lucky,:] )
        
        x = torch.cat(segment_list, dim=2)
        
        idx = torch.randperm(x.shape[2])
        x = x[:,:,idx,:].unsqueeze(3)
        
        return x, None
"""

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
