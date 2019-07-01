import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
from torch.autograd import Variable
#import torch.jit

#torch.jit.ScriptModule
#torch.jit.trace(nn.Sigmoid(), torch.rand(1,1,1))

class CutoffScaleSigmoid(nn.Module):
    __constants__ = ['cutoff', 'mean', 'momentum']
    def __init__(self, cutoff=1., mean=0.5, momentum=0.1):
        super(CutoffScaleSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.running_mean = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.cutoff = cutoff
        self.mean = mean
        self.momentum = momentum

    #@torch.jit.script_method
    def forward(self, x):
        y = ((1.+self.cutoff)*self.sigmoid(x)-self.cutoff).clamp(0.)
        y_mean = y.mean().detach()
        if self.training:
            self.running_mean += self.momentum * (y_mean - self.running_mean)
            ret_val = (y * self.mean/y_mean).clamp(0.,1.)
        else:
            ret_val = (y * self.mean/self.running_mean).clamp(0.,1.)
        return ret_val






def r_d_max_func(itr):
    "Default max r and d provider as recommended in paper."
    if itr < 5000:
        return 1, 0
    if itr < 40000:
        r_max = 2 / 35000 * (itr - 5000) + 1
    else:
        r_max = 3
    if itr < 25000:
        d_max = 5 / 20000 * (itr - 5000)
    else:
        d_max = 5
    return r_max, d_max


class BatchRenorm(nn.Module):
    __constants__ = ['eps','momentum', 'update_interval', 'noise']
    def __init__(self, num_features, update_interval, eps=1e-5, momentum=0.1, noise=0.3):
        super(BatchRenorm, self).__init__()
        self.update_interval = update_interval-1
        self.eps = eps
        self.momentum = momentum
        self.noise = noise
        #self.register_buffer('itr', torch.zeros(1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_sigma', torch.ones(num_features))

    #@torch.jit.script_method
    def forward(self, input, iteration:int):
        shp = [input.shape[1],] + [1,1][:len(input.shape)-2]
        
        if self.training:
            # flatten to (batch*height*width, channels)
            input_flat = input.transpose(-1, 1).contiguous().view((-1, input.shape[1])) + self.eps
    
            # Calculate batch/norm statistics
            # mean_b = input_flat.mean(0).view(shp) - self.noise * torch.rand(shp).cuda()
            #sigma_b = input_flat.std(0).view(shp) + self.eps #.expand_as(input) + self.eps
            mean_b = input_flat.sum(0)/input_flat.shape[0] #- self.noise * torch.rand(input_flat.shape[1]).cuda()
            sigma_b = (((input_flat - mean_b) ** 2).sum(0)/input_flat.shape[0] + self.eps).sqrt()

            bn = (input - mean_b.view(shp)) / sigma_b.view(shp)

            #r_max, d_max = r_d_max_func(self.itr.item())
            #r = (sigma_b.detach()/self.running_sigma).clamp(1/r_max, r_max)
            #d = ((mean_b.detach()-self.running_mean) / self.running_sigma).clamp(-d_max, d_max)
            
            #bn = bn * r.view(shp) + d.view(shp)
            
            #bn += 0.3*torch.rand_like(bn).cuda() - 0.05
            #bn -= 0.3*torch.rand(shp).cuda()
            #randn(shp, device=input.device)
    
            # Update moving stats
            if iteration == self.update_interval: #self.n == self.update_interval:
                #self.n *= 0
                #self.steps += 1
                #self.itr += 1
                self.running_mean += self.momentum * (mean_b.detach() - self.running_mean)
                self.running_sigma += self.momentum * (sigma_b.detach() - self.running_sigma)
        else:
            bn = (input - self.running_mean.view(shp)) / self.running_sigma.view(shp)

        return bn

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    #@torch.jit.script_method
    def forward(self, x, iteration:int):
        lambda_ = 1 - 0.95 ** (iteration+1)
        return self.sigmoid(lambda_*x)

class CutoffSigmoid(nn.Module):
    __constants__ = ['cutoff', 'scale']
    def __init__(self, cutoff=1., scale=2.4):
        super(CutoffSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.cutoff = cutoff
        self.scale = scale / (1 - 0.95**3)

    #@torch.jit.script_method
    def forward(self, x, iteration:int):
        lambda_ = (1 - 0.95 ** (iteration+1)) * self.scale / (1.+self.cutoff)
        y = ((1.+self.cutoff)*self.sigmoid(lambda_*x)-self.cutoff).clamp(0.)
        return y

class CutoffSigmoid2(nn.Module):
    __constants__ = ['cutoff', 'offset']
    def __init__(self, cutoff=0.2, offset=-0.4):
        super(CutoffSigmoid2, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.cutoff = 0.5
        self.offset = 0.

    #@torch.jit.script_method
    def forward(self, x, iteration:int):
        lambda_ = 1 - 0.95 ** (iteration+1)
        y = self.sigmoid(lambda_*x + self.offset)
        y = y - (y < self.cutoff).float() * y
        return y

class LeakyCutoffSigmoid(nn.Module):
    __constants__ = ['cutoff']
    def __init__(self, cutoff=1.):
        super(LeakyCutoffSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.cutoff = cutoff

    #@torch.jit.script_method
    def forward(self, x, iteration:int):
        lambda_ = 1 - 0.95 ** (iteration+1)
        x *= lambda_
        y = torch.max(  (1.+self.cutoff)*self.sigmoid(x)-self.cutoff , x * 0.0001 + 0.01   ).clamp(0., 1.)
        return y


class NormedActivation(nn.Module):
    def __init__(self, num_features, update_interval, activation, momentum):
        super(NormedActivation, self).__init__()
        #self.weight = nn.Parameter(torch.randn(num_features).view(1,num_features,1))
        self.batcnnorm = BatchRenorm(num_features=num_features, update_interval=update_interval, momentum=momentum)
        self.activation = activation

    #@torch.jit.script_method
    def forward(self, x, iteration:int):
        x = self.batcnnorm(x, iteration)
        y = self.activation(x, iteration)
        return y


"""
def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = torch.mean(X, dim=0)
        # mini-batch variance
        variance = torch.mean((X - mean) ** 2, dim=0)
        # normalize
        X_hat = (X - mean) * 1.0 / torch.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = torch.mean(X, dim=(0, 2, 3))
        # mini-batch variance
        variance = torch.mean((X - mean.view((1, C, 1, 1))) ** 2, dim=(0, 2, 3))
        # normalize
        X_hat = (X - mean.view((1, C, 1, 1))) * 1.0 / torch.sqrt(variance.view((1, C, 1, 1)) + eps)
        # scale and shift
        out = gamma.view((1, C, 1, 1)) * X_hat + beta.view((1, C, 1, 1))

    return out



def r_d_max_func(itr):
    if itr < 5000:
        return 1, 0
    if itr < 40000:
        r_max = 2 / 35000 * (itr - 5000) + 1
    else:
        r_max = 3
    if itr < 25000:
        d_max = 5 / 20000 * (itr - 5000)
    else:
        d_max = 5
    return r_max, d_max

class BatchRenorm(nn.Module):
    def __init__(self, num_features, affine=True, r_d_func=r_d_max_func, eps=1e-5, momentum=0.1):
        super(BatchRenorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.r_d_func = r_d_func
        self.eps = eps
        self.momentum = momentum
        if affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_sigma', torch.ones(num_features))
        self.reset_parameters()
        self.not_initialized = True

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_sigma.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.size(1) != self.running_mean.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.num_features))

    def forward(self, input, do_update=True):
        #_, C, H, W = input.size()
        if self.not_initialized:
            self.C = input.shape[1]
            if len(input.shape)==4:
                self.mean_shp = (0,2,3)
                self.sigma_shp = (1,self.C,1,1)
                self.expand_shp = input.shape[1:]
                self.d_shp = (-1,1,1)
            else:
                self.mean_shp = (0,2)
                self.sigma_shp = (1,self.C,1)
                self.expand_shp = input.shape[1:]
                self.d_shp = (-1,1)
            self.running_mean = torch.mean(input.data, dim=self.mean_shp)
            self.running_sigma = torch.sqrt( torch.mean((input.data - self.running_mean.view(self.sigma_shp)) ** 2, dim=self.mean_shp) + self.eps)
            self.not_initialized = False

        #if do_update and self.training:
        #    mean = torch.mean(input.data, dim=self.mean_shp)
        #    sigma = torch.sqrt( torch.mean((input.data - mean.view(self.sigma_shp)) ** 2, dim=self.mean_shp) + self.eps)
        #    self.running_mean = self.running_mean + self.momentum * (mean-self.running_mean)
        #    self.running_sigma = self.running_sigma + self.momentum * (sigma-self.running_sigma)
        #bn = (input - self.running_mean.view(self.sigma_shp)) * 1.0 / (self.running_sigma.view(self.sigma_shp)+self.eps)

        if self.training:
            mean = torch.mean(input, dim=self.mean_shp)
            sigma = torch.sqrt( torch.mean((input - mean.view(self.sigma_shp)) ** 2, dim=self.mean_shp) + self.eps)
            bn = (input - mean.view(self.sigma_shp)) * 1.0 / (sigma.view(self.sigma_shp) + self.eps)
            
            r_max, d_max = 1, 0 #self.r_d_func(itr)
            r = (sigma.data/(self.running_sigma+self.eps)).clamp(1/r_max, r_max)
            d = ((mean.data-self.running_mean) / (self.running_sigma+self.eps)).clamp(-d_max, d_max)
            if do_update:
                self.running_mean = self.running_mean + self.momentum * (mean.data-self.running_mean)
                self.running_sigma = self.running_sigma + self.momentum * (sigma.data-self.running_sigma)
            bn = bn * r.view(self.d_shp).expand(self.expand_shp) + d.view(self.d_shp).expand(self.expand_shp)
        else:
            bn = (input - self.running_mean.view(self.sigma_shp)) * 1.0 / (self.running_sigma.view(self.sigma_shp)+self.eps)

        if self.affine:
            return self.weight.view(self.sigma_shp) * bn + self.bias.view(self.sigma_shp)
        else:
            return bn

    def activate(self, input, do_update=True):
        _, C, H, W = input.size()

        if self.training:
            mean = torch.mean(input, dim=(0, 2, 3))
            sigma = torch.sqrt( torch.mean((input - mean.view((1, C, 1, 1))) ** 2, dim=(0, 2, 3)) + self.eps)
            bn = (input - mean.view(1, C, 1, 1)) * 1.0 / (sigma.view(1, C, 1, 1) + self.eps)
            if do_update:
                self.running_mean = self.running_mean + self.momentum * (mean.data-self.running_mean)
                self.running_sigma = self.running_sigma + self.momentum * (sigma.data-self.running_sigma)
        else:
            bn = (input - self.running_mean.view(1, C, 1, 1)) * 1.0 / (self.running_sigma.view(1, C, 1, 1)+self.eps)

        if self.affine:
            return self.weight.view((1, C, 1, 1)) * bn + self.bias.view((1, C, 1, 1))

        return bn


    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum})'.format(name=self.__class__.__name__, **self.__dict__))
"""
