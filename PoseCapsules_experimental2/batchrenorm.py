import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
from torch.autograd import Variable

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
        _, C, H, W = input.size()

        if self.training:
            mean = torch.mean(input, dim=(0, 2, 3))
            sigma = torch.sqrt( torch.mean((input - mean.view((1, C, 1, 1))) ** 2, dim=(0, 2, 3)) + self.eps)
            bn = (input - mean.view(1, C, 1, 1)) * 1.0 / (sigma.view(1, C, 1, 1) + self.eps)
            
            r_max, d_max = 1, 0 #self.r_d_func(itr)
            r = (sigma.data/(self.running_sigma+self.eps)).clamp(1/r_max, r_max)
            d = ((mean.data-self.running_mean) / (self.running_sigma+self.eps)).clamp(-d_max, d_max)
            if do_update:
                self.running_mean = self.running_mean + self.momentum * (mean.data-self.running_mean)
                self.running_sigma = self.running_sigma + self.momentum * (sigma.data-self.running_sigma)
            bn = bn * r.view(-1,1,1).expand(C,H,W) + d.view(-1,1,1).expand(C,H,W)
        else:
            bn = (input - self.running_mean.view(1, C, 1, 1)) * 1.0 / (self.running_sigma.view(1, C, 1, 1)+self.eps)

        if self.affine:
            return self.weight.view((1, C, 1, 1)) * bn + self.bias.view((1, C, 1, 1))
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

    """
    # batch_size, output_dim, h, (dim_x, dim_y)
    def forward2(self, input, do_update=True):
        _, C, H, W = input.size()
        dim = (0, 2, 3)
        output_dim = (1, C, 1, 1)

        if self.training:
            mean = torch.mean(input.norm(p=2, dim=2, keep_dim=True), dim=dim)
            sigma = torch.sqrt( torch.mean((input - mean.view(output_dim)) ** 2, dim=dim) + self.eps)
            bn = (input - mean.view(output_dim)) * 1.0 / sigma.view(output_dim)
            
            r_max, d_max = 1, 0 #self.r_d_func(itr)
            r = (sigma.data/self.running_sigma).clamp(1/r_max, r_max)
            d = ((mean.data-self.running_mean) / self.running_sigma).clamp(-d_max, d_max)
            if do_update:
                self.running_mean = self.running_mean + self.momentum * (mean.data-self.running_mean)
                self.running_sigma = self.running_sigma + self.momentum * (sigma.data-self.running_sigma)
            bn = bn * r.view(-1,1,1).expand(C,H,W) + d.view(-1,1,1).expand(C,H,W)
        else:
            bn = (input - self.running_mean.view(1, C, 1, 1)) * 1.0 / self.running_sigma.view(1, C, 1, 1)
            
        return self.weight.view((1, C, 1, 1)) * bn + self.bias.view((1, C, 1, 1))
    """

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum})'.format(name=self.__class__.__name__, **self.__dict__))

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
"""
