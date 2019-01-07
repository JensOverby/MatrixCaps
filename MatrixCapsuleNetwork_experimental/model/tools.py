'''
Created on Oct 1, 2018

@author: jens
'''

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_2d_sample(sample):
    sample_np = sample.numpy()
    x = sample_np[:, 0]
    y = sample_np[:, 1]
    plt.scatter(x, y)

def get_k_likelihoods(X, mu, var):
    """
    Compute the densities of each data point under the parameterised gaussians.
    
    :param X: design matrix (examples, features)
    :param mu: the component means (K, features)
    :param var: the component variances (K, features)
    :return: relative likelihoods (K, examples)
    """
    
    if var.data.eq(0).any():
        raise Exception('variances must be nonzero')
    
    # get the trace of the inverse covar. matrix
    covar_inv = 1. /  var # (K, features)
    
    # compute the coefficient
    det = (2 * np.pi * var).prod(dim=1) # (K)
    coeff = 1. / det.sqrt() # (K)
    
    # tile the design matrix `K` times on the batch dimension 
    K = mu.size(0)
    X = X.unsqueeze(0).repeat(K, 1, 1)
    
    # calculate the exponent
    a = (X - mu.unsqueeze(1)) # (K, examples, features)
    exponent = a ** 2 @ covar_inv.unsqueeze(2)
    exponent = -0.5 * exponent
    
    # compute probability density
    P = coeff.view(K, 1, 1) * exponent.exp()
    
    # remove final singleton dimension and return
    return P.squeeze(2)

def get_density(mu, var, N=50, X_range=(0, 5), Y_range=(0, 5)):
    """ Get the mesh to compute the density on. """
    X = np.linspace(*X_range, N)
    Y = np.linspace(*Y_range, N)
    X, Y = np.meshgrid(X, Y)
    
    # get the design matrix
    points = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    points = Variable(torch.from_numpy(points).float())
    
    # compute the densities under each mixture
    P = get_k_likelihoods(points, mu, var)

    # sum the densities to get mixture density
    Z = torch.sum(P, dim=0).data.numpy().reshape([N, N])
    
    return X, Y, Z

def plot_density(X, Y, Z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.inferno)

    z_max = Z.max()
    z_min = Z.min()

    offset = (z_max-z_min)*0.2

    cset = ax.contourf(X, Y, Z, zdir='z', offset=z_min-offset, cmap=cm.inferno)

    # adjust the limits, ticks and view angle
    ax.set_zlim(z_min,z_max)
    #ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()


    """
    def debugPrepare(self, V, a_):
        off = 3
        self.b = 1
        self.hh = 2
        self.Cww = 4
        self.C = 1
        self.iteration = 10
        self.max_x = None
        V = V[self.b,:self.Bkk,:self.Cww,off:off+self.hh].unsqueeze(0)
        a_ = a_[self.b,:self.Bkk,:self.Cww].unsqueeze(0)
        self.beta_v.data = self.beta_v[:,:self.C,:,:]
        self.beta_a.data = self.beta_a[:,:self.C,:]
        return V, a_

    def debugDo(self, mu, sigma_square, R, V):
        _mu = mu.data.cpu().view(self.Cww, self.hh)
        _si = sigma_square.data.cpu().view(self.Cww, self.hh)
        if self.max_x is None:
            tmp = (R.data.cpu() * V.data.cpu()).view(self.Bkk, self.Cww, self.hh)
            self.max_x = tmp[:,:,0].max()
            self.min_x = tmp[:,:,0].min()
            self.max_y = tmp[:,:,1].max()
            self.min_y = tmp[:,:,1].min()
        tools.plot_density(*tools.get_density(_mu, _si, N=100, X_range=(self.min_x, self.max_x), Y_range=(self.min_y, self.max_y)))
    """
