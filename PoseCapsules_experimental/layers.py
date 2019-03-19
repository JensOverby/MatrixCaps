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


"""
def calc_same_padding(input_, kernel=1, stride=1, dilation=1, transposed=False):
    if transposed:
        return (dilation*(kernel-1) + 1) // 2 - 1, input_ // (1./stride)
    else:
        return (dilation*(kernel-1) + 1) // 2, input_ // stride
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
    #layer_list = OrderedDict()
    model = [] #nn.Sequential()
    for i, layer_size in enumerate(layer_sizes, 1):
        name = 'layer' + str(i)
        layer = nn.Linear(last_layer_size, layer_size)
        nn.init.normal_(layer.weight.data, mean=mean, std=std)
        nn.init.constant_(layer.bias.data, val=bias)
        model.append((name, layer))
        #layer_list[name] = layer
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

class PrimToCaps(nn.Module):
    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2).contiguous()                   # batch_size, input_dim, dim_x, dim_y, input_atoms
        return x.view(x.size(0), -1, x.size(-1), 1, 1)                 # batch_size, input_dim*dim_x*dim_y, input_atoms, 1, 1

class ConvToPrim(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)

class MatrixToOut(nn.Module):
    def __init__(self, output_dim):
        super(MatrixToOut, self).__init__()
        self.output_dim = output_dim
    def forward(self, x):
        return x[0][...,:self.output_dim]

class PrimToMatrixPrim(nn.Module):
    def forward(self, x):
        sh = x.size()
        return x.view(sh[0], sh[1]*sh[2], sh[3], sh[4])

class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MyConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) #.cuda(self.device)
        nn.init.normal_(self.weight.data, mean=0,std=0.1)
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x_orig):
        x_sh = x.size() # batch_size*input_dim, input_atoms, dim_x, dim_y
        w = int((x_sh[2] - self.kernel_size) / self.stride) + 1
        pose_list = []
        for i in range(w):
            for j in range(w):
                tile = x[:, :, self.stride * i:self.stride * i + self.kernel_size, self.stride * j:self.stride * j + self.kernel_size]
                tile = self.weight[None,...] * tile[:,None,...]
                tile = tile.view(x_sh[0], self.out_channels,-1).sum(-1)
                pose_list.append( tile )
        poses = torch.stack(pose_list, dim=-1)
        poses = poses.view(x_sh[0], self.out_channels, w, w)
        return poses

class CapsuleLayer(nn.Module):
    
    """
    output_dim:    number of classes
    """
    
    def __init__(self, output_dim, h, num_routing, voting):
        super(CapsuleLayer, self).__init__()
        self.not_initialized = True
        self.output_dim = output_dim
        self.h = h
        self.num_routing = num_routing
        self.voting = voting

    def init(self, x):
        self.do_sort = 0
        if self.voting.get('sort') != None:
            self.do_sort = self.voting['sort']
        self.do_add = False
        if self.voting.get('add') != None:
            self.do_add = True
        self.type = 1

        if self.voting['type'] == 'standard':
            self.weight = nn.Parameter(torch.Tensor(x.size(1), x.size(2), self.output_dim * self.h, 1, 1))
            nn.init.normal_(self.weight.data, mean=0,std=0.1)      #input_dim, input_atoms, output_dim*h
            self.bias = nn.Parameter(torch.Tensor(self.output_dim, self.h, 1, 1))
            nn.init.constant_(self.bias.data, val=0.1)
            self.type = 0
        elif self.voting['type'] == 'matrix':
            self.weight = nn.Parameter(torch.Tensor(1, x[0].size(1), self.output_dim, self.h, self.h)) 
            nn.init.normal_(self.weight.data, mean=0,std=0.1)      #input_dim, input_h, input_h, output_dim, output_h, output_h
            self.beta_v = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1,1))
            self.beta_a = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1))
            if self.voting.get('lambda') == None:
                self.lambda_ = 1e-3
            else:
                self.lambda_ = self.voting['lambda']
            self.step = 5e-4
            self.type = 2
        elif self.voting['type'] == 'prim_matrix':
            self.capsules_pose = nn.Conv2d(in_channels=x.size(1), out_channels=self.output_dim*self.h*self.h, kernel_size=1, stride=1, bias=True)
            self.capsules_activation = nn.Conv2d(in_channels=x.size(1), out_channels=self.output_dim, kernel_size=1, stride=1, bias=True)
            self.type = 3
        elif self.voting['type'] == 'Conv2d':
            x = x.data
            self.conv = nn.Conv2d(in_channels=x.size(2),
                                           out_channels=self.output_dim * self.h,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'],
                                           padding=self.voting['padding'],
                                           bias=False)
            self.bias = nn.Parameter(torch.Tensor(self.output_dim, self.h, 1, 1))
            nn.init.constant_(self.bias.data, val=0.1)
            if self.do_sort:
                #self.avg_conv_weight = torch.Tensor(self.conv.weight.size()).fill_(1/(self.conv.kernel_size[0]*self.conv.kernel_size[1])).cuda(self.device)
                w = self.conv(x.data.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))).size(-1)
                self.add = nn.Parameter(torch.Tensor([[i / w, j / w] for i in range(w) for j in range(w)]).permute(1,0).view(-1,w,w), requires_grad=False)
                self.scale = nn.Parameter(torch.Tensor([50.]))
        elif self.voting['type'] == 'ConvTranspose2d':
            self.conv = nn.ConvTranspose2d(in_channels=x.size(2),
                                           out_channels=self.output_dim * self.h,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'],
                                           padding=self.voting['padding'],
                                           bias=False)#.cuda(self.device)
            self.bias = nn.Parameter(torch.Tensor(self.output_dim, self.h, 1, 1))
            nn.init.constant_(self.bias.data, val=0.1)
        elif self.voting['type'] == 'experimental':
            self.conv = MyConv(in_channels=x.size(2),
                                           out_channels=self.output_dim * self.h,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'])#.cuda(self.device)
        else:
            raise NotImplementedError('Convolutional type not recognized. Must be: Conv2d, Conv3d, ConvTranspose2d, or ConvTranspose3d"')

        if self.type == 1:
            nn.init.normal_(self.conv.weight.data, mean=0,std=0.1)

        self.not_initialized = False


    def forward(self, x):
        if self.not_initialized:
            self.init(x)
            
        if self.type == 0:
            x.unsqueeze_(3)                                         # batch_size, input_dim, input_atoms, 1, dim_x, dim_y
            tile_shape = list(x.size())
            tile_shape[3] = self.output_dim * self.h     # batch_size, input_dim, input_atoms, output_dim*h, dim_x, dim_y
            x = x.expand(tile_shape)                                # batch_size, input_dim, input_atoms, output_dim*h, dim_x, dim_y
            x = torch.sum(x * self.weight, dim=2)                  # batch_size, input_dim, output_dim*h

            votes = x.view(x.size(0), -1, self.output_dim, self.h, x.size(-2), x.size(-1))
            biases_replicated = self.bias.repeat([1,1,x.size(-2),x.size(-1)])
            logit_shape = list(votes.size())
            logit_shape.pop(3)                                          # batch_size, input_dim, output_dim, dim_x, dim_y
            return dynamic_routing(votes=votes, biases=biases_replicated, logit_shape=logit_shape, num_routing=self.num_routing)
        
        elif self.type == 1:
            x_sh = x.size()                                             # batch_size, input_dim, input_atoms, dim_x, dim_y
            x = x.view(x_sh[0]*x_sh[1], x_sh[2], x_sh[3], x_sh[4])  # batch_size*input_dim, input_atoms, dim_x, dim_y

            if self.do_sort:
                """ Create Keys and center these with a mean of zero """
                shp = self.conv.weight.data.shape
                mean_weight = self.conv.weight.data.view(shp[0],shp[1],-1).mean(2, keepdim=True).unsqueeze(-1).expand(self.conv.weight.size())
                corr_weight = (self.conv.weight.data - mean_weight)
                del mean_weight
                
                """ Normalize keys, so high contrast keys don't contribute more than low contrast keys """
                norm = corr_weight.view(shp[0],shp[1],-1).norm(p=1, dim=2, keepdim=True).unsqueeze(-1)
                norm_weight = corr_weight / norm
                del corr_weight
                del norm
                
                """ Center x with a mean of zero """
                mean_x = x.data.view(x.size(0),x.size(1),-1).mean(dim=2, keepdim=True).unsqueeze(-1)
                corr_x = x.data - mean_x
                del mean_x

                """ Convolve x """
                x = self.conv(x)
                
                """ Convolve coor_x - Try to fit keys """
                corr_x = F.conv2d(corr_x, norm_weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, 1)
                del norm_weight
                
                #a_sort = corr_x.norm(p=2, dim=1).view(x_sh[0],-1).sort(1, descending=True)[1]
                a_sort = corr_x.view(x.size(0), self.output_dim, self.h, -1).norm(p=1, dim=2).norm(p=2, dim=1).sort(1, descending=True)[1]
                #a_sort = corr_x.view(x_sh[0], self.output_dim, self.h, -1).norm(p=1, dim=2).max(dim=1)[0].sort(1, descending=True)[1]
                #a_sort = corr_x.view(x_sh[0], self.output_dim, self.h, -1).norm(p=1, dim=2).view(x_sh[0],-1).sort(1, descending=True)[1]
                del corr_x
                a_sort = a_sort[:,None,:].repeat(1,x.shape[1],1)
                #a_sort = a_sort.view(x_sh[0],self.output_dim,1,-1).repeat(1,1,self.h,1).view(x.shape[0],x.shape[1],-1)
                x = x.view(x.size(0), self.output_dim, -1, x.size(-2), x.size(-1))

                if self.do_add:
                    x[:,:,:2,:,:] = x[:,:,:2,:,:] + self.add * self.scale
                #x = x.view(x.size(0),x.size(1)*x.size(2),-1).gather(2, a_sort)[:,:,:self.do_sort*2,None]
                #idx = torch.randperm(self.do_sort*2)
                #x = x[:,:,idx,:][:,:,:self.do_sort,:]
                x_sorted = x.view(x.size(0),x.size(1)*x.size(2),-1).gather(2, a_sort)
                
                best = x_sorted[:,:,:self.do_sort,None]
                rest = x_sorted[:,:,self.do_sort:,None]
                idx_lucky = torch.randperm(rest.size(2))[:int(self.do_sort*0.25)]
                x = torch.cat([best,rest[:,:,idx_lucky,:]], dim=2)
                
                #x = x_sorted[:,:,:self.do_sort,None]
                
                #x = x_sorted[:,:,:self.do_sort*2,None]
                #idx = torch.randperm(self.do_sort*2)
                #x = x[:,:,idx,:][:,:,:self.do_sort,:]


                
                del a_sort
            else:
                x = self.conv(x)                                        # batch_size*input_dim, output_dim*h, out_dim_x, out_dim_y

            votes = x.view(x_sh[0], -1, self.output_dim, self.h, x.size(-2), x.size(-1))
            biases_replicated = self.bias.repeat([1,1,x.size(-2),x.size(-1)])
            logit_shape = list(votes.size())
            logit_shape.pop(3)                                          # batch_size, input_dim, output_dim, dim_x, dim_y
            return dynamic_routing(votes=votes, biases=biases_replicated, logit_shape=logit_shape, num_routing=self.num_routing)
        
        elif self.type == 2:
            poses, activations = x
            # poses:        batch_size, input_dim, hh
            # activations:  batch_size, input_dim
            shp = poses.size()
            poses = poses.view(shp[0], shp[1], 1, self.h, self.h)
            
            votes = self.weight @ poses # batch_size, input_dim, output_dim, output_h, output_h
            votes = votes.view(shp[0], shp[1], self.output_dim, self.h*self.h)
            activations = activations.unsqueeze(-1).repeat(1, 1, self.output_dim) # batch_size, input_dim, output_dim
    
            poses, activations = EM_routing(self.lambda_, votes, activations, self.beta_v, self.beta_a, self.num_routing)
            
            if self.training and self.lambda_ < 1:
                self.lambda_ += self.step
            
            return poses.squeeze(1), activations
        
        else:
            poses = self.capsules_pose(x)
            sh = poses.size()
            poses = poses.view(sh[0], self.output_dim, -1, sh[2], sh[3]).permute(0, 1, 3, 4, 2).contiguous()
            activations = self.capsules_activation(x)
            #activations = poses.norm(p=2, dim=-1)
            activations = torch.sigmoid(activations)
            return poses.view(sh[0], -1, self.h*self.h), activations.view(sh[0], -1)

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
            
    return activation                                               # batch_size, output_dim, h, (dim_x, dim_y)

def _squash(input_tensor):
    norm = torch.norm(input_tensor, p=2, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def EM_routing(lambda_, V, a_, beta_v, beta_a, num_routing):
    # routing coefficient
    shp = V.size()
    R = Variable(torch.ones([shp[0], shp[1], shp[2]], device=V.device), requires_grad=False) / shp[2]

    for i in range(num_routing):
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
        cost = (beta_v + log_sigma.view(shp[0],shp[2],-1,shp[-1])) * sum_R.view(shp[0], shp[2],-1,1)
        a = torch.sigmoid(lambda_ * (beta_a - cost.sum(-1)))
        a = a.view(shp[0], shp[2])

        # E-step
        if i != num_routing - 1:
            ln_p_j_h = -V_minus_mu_sqr / (2 * sigma_square) - log_sigma - 0.5*ln_2pi
            p = torch.exp(ln_p_j_h)
            ap = a[:,None,:] * p.sum(-1)
            R = Variable(ap / (torch.sum(ap, 2, keepdim=True) + eps) + eps, requires_grad=False) # detaches from graph

    return mu, a
