import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from batchrenorm import BatchRenorm

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
        return x[0][...,:self.output_dim]


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
            row = torch.zeros(w)
            step = math.pi/(w-1)
            for i in range(w):
                row[i] = math.sin(i*step)
            """


            #self.m = torch.arange(w*w).float().view(w,w)

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
        self.beta_v = nn.Parameter(torch.randn(output_dim).view(1,output_dim,1,1))
        self.beta_a = nn.Parameter(torch.randn(output_dim).view(1,output_dim,1))
        
    def forward(self, votes): # (b, Bkk, Cww, hh)
        shp = votes.shape

        """ If previous was ConvVector2d """
        if len(shp) == 6: # batch, input_dim, output_dim, h, out_dim_x, out_dim_y
            votes = votes.permute(0,1,2,4,5,3).contiguous()
            votes = votes.view(shp[0], shp[1], shp[2]*shp[4]*shp[5], shp[3])
            shp = votes.shape
        
        V = votes[...,:shp[3]-1]
        a_ = votes[...,shp[3]-1:]
        w = int(math.sqrt(shp[2] / self.output_dim))

        # routing coefficient
        if V.is_cuda:
            R = Variable(torch.ones(shp[:3]), requires_grad=False).cuda() / self.output_dim
        else:
            R = Variable(torch.ones(shp[:3]), requires_grad=False) / self.output_dim

        for i in range(self.num_routing):
            # M-step
            R = (R * a_).unsqueeze(-1)
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R).unsqueeze(1)
            V_minus_mu_sqr = (V - mu) ** 2
            sigma_square = ((R * V_minus_mu_sqr).sum(1) / sum_R).unsqueeze(1)

            """
            beta_v: Bias for log probability of sigma ("standard deviation")
            beta_a: Bias for offsetting
            
            In principle, beta_v and beta_a are only for learning regarding "activations".
            Just like "batch normalization" it has both scaling and bias.
            Votes are routed by the learned weight self.W.
            """
            log_sigma = torch.log(sigma_square.sqrt()+eps)
            cost = (self.beta_v + log_sigma.view(shp[0],self.output_dim,-1,shp[3]-1)) * sum_R.view(shp[0], self.output_dim,-1,1)
            a = torch.sigmoid(lambda_ * (self.beta_a - cost.sum(-1)))
            a = a.view(shp[0], shp[2])

            # E-step
            if i != self.iteration - 1:
                ln_p_j_h = -V_minus_mu_sqr / (2 * sigma_square) - log_sigma - 0.5*ln_2pi
                p = torch.exp(ln_p_j_h)
                ap = a[:,None,:] * p.sum(-1)
                R = Variable(ap / (torch.sum(ap, 2, keepdim=True) + eps) + eps, requires_grad=False) # detaches from graph

        mu_a = torch.cat([mu.view(shp[0], self.output_dim, w, w, -1), a.view(shp[0], self.output_dim, w, w, 1)], dim=-1) # b, C, w, w, hh
        return mu_a


class MaxRoutePool(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxRoutePool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        self.not_initialized = True
        
    def forward(self, x):
        """
        votes: batch_size, input_dim, output_dim, h, out_dim_x, out_dim_y
        route: batch_size, input_dim, output_dim, (dim_x, dim_y)
        """

        """ If previous was VectorRouting """
        if len(x) == 3:
            votes, route = x[2], x[1]
        else:
            votes, route = x[0], x[1]
        
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
    def __init__(self, out_sz):
        super(MaxRouteReduce, self).__init__()
        self.out_sz = int(0.75*out_sz)
        self.rnd_out_sz = out_sz - self.out_sz
        
    def forward(self, x):

        """ Calculate routing coefficients """
        # votes: batch, input_dim, output_dim, h, dim_x, dim_y
        # route: batch, input_dim, output_dim, dim_x, dim_y
        
        #capsule_routing = x[1].max(dim=1)[0] # batch, output_dim, dim_x, dim_y
        #capsule_routing = x[1].sum(dim=1) # batch, output_dim, dim_x, dim_y
        """ If previous was VectorRouting """
        if len(x) == 3:
            votes, route = x[2], x[1]
        else:
            votes, route = x[0], x[1]
        
        v_shp = votes.shape
        #capsule_routing = x[1].view(-1, x_sh[1], x_sh[-2], x_sh[-1])
        capsule_routing = route.sum(dim=2) # batch, input_dim, dim_x, dim_y

        """ Rank routing coefficients """
        a_sort = capsule_routing.view(v_shp[0], v_shp[1], -1).sort(2, descending=True)[1] # batch, input_dim, dim_x*dim_y
        a_sort = a_sort[:,:,None,None,:].repeat(1,1,v_shp[2],v_shp[3],1)

        x_sorted = votes.view_as(a_sort).gather(4, a_sort)
        best = x_sorted[:,:,:,:,:self.out_sz]
        rest = x_sorted[:,:,:,:,self.out_sz:]
        idx_lucky = torch.randperm(rest.size(4))[:self.rnd_out_sz]
        x_sorted = torch.cat([best,rest[:,:,:,:,idx_lucky]], dim=4)

        idx = torch.randperm(self.out_sz+self.rnd_out_sz)
        x_sorted = x_sorted[:,:,:,:,idx].unsqueeze(-1)        # batch, input_dim, output_dim, h, dim_x, dim_y

        return x_sorted, None # is None, to force bias detach

        #x = x_sorted.permute(0, 1, 3, 4, 2).contiguous()    # batch_size, input_dim, output_dim, dim_x, dim_y, input_atoms
        #return (x.view(x.size(0), -1, x.size(-1), 1, 1))    # batch_size, input_dim*dim_x*dim_y, input_atoms, 1, 1

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
