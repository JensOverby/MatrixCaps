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

        shp = R.shape
        if len(shp) == 5:
            capsule_routing = R.data.view(shp[0]*shp[1], shp[2], -1)
        else:
            capsule_routing = R.data.view(shp[0],shp[1],-1) #.view(shp[0], shp[1], -1)
        
        if self.lifetime:
            """ Rank routing coefficients """
            capsule_routing_sum = capsule_routing.sum(dim=-1) # batch-size*input_dim, output_dim
            order = capsule_routing_sum.sort(1, descending=True)[1]
            ranks = (-order).sort(1, descending=True)[1]
        
            """ Winning frequency """
            if self.masked_freq:
                masked_routing = ((ranks < self.k).float() * capsule_routing_sum).sum(dim=0)
                freq = masked_routing / masked_routing.sum()
            else:
                win_counts = (ranks < self.k).sum(dim=0)
                freq = win_counts.float() / (self.k*ranks.shape[0]) # output_dim
        else:
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
        if sparse is None:
            self.beta_a = nn.Parameter(torch.randn(self.output_dim).view(1,self.output_dim,1))

        self.stat = stat
        if stat is not None:
            for _ in range(4):
                self.stat.append(0.)
        
    def forward(self, x): # (b, Bkk, Cww, h)
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
