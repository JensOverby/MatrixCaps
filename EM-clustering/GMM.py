import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from scipy.stats import multivariate_normal
np.cat = np.concatenate

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from tqdm import tqdm_notebook as tqdm
from torch.distributions import Normal


torch.__version__

def sample(mu, var, nb_samples=500):
    """
    Return a tensor of (nb_samples, features), sampled
    from the parameterized gaussian.
    
    :param mu: torch.Tensor of the means
    :param var: torch.Tensor of variances (NOTE: zero covars.)
    """
    out = []
    for i in range(nb_samples):
        out += [
            torch.normal(mu, var.sqrt()).unsqueeze(0)
        ]
    return torch.cat(out, dim=0)

# generate some clusters
cluster1 = sample(
    torch.Tensor([2.5, 2.5]),
    torch.Tensor([1.2, .8])
)

cluster2 = sample(
    torch.Tensor([7.5, 7.5]),
    torch.Tensor([.75, .5])
)

cluster3 = sample(
    torch.Tensor([8, 1.5]),
    torch.Tensor([.6, .8])
)

def plot_2d_sample(sample):
    sample_np = sample.numpy()
    x = sample_np[:, 0]
    y = sample_np[:, 1]
    plt.scatter(x, y)
    
# create the dummy dataset, by combining the clusters.
X = torch.cat([cluster1, cluster2, cluster3])
plot_2d_sample(X)

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

def plotXYZ(XYZ):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    for i in range(XYZ.shape[0]):
        for j in range(XYZ.shape[1]):
            xs, ys, zs = XYZ[i,j,:]
            ax.scatter(xs, ys, zs, c='r', marker='o')
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()

#>>> points.shape
#torch.Size([10000, 2])
#>>> P.shape
#torch.Size([3, 10000])    

def get_density(mu, var, pi, N=50, X_range=(0, 5), Y_range=(0, 5)):
    """ Get the mesh to compute the density on. """
    X = np.linspace(*X_range, N)
    Y = np.linspace(*Y_range, N)
    X, Y = np.meshgrid(X, Y)
    
    # get the design matrix
    points = np.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    points = Variable(torch.from_numpy(points).float())
    
    # compute the densities under each mixture
    P = get_k_likelihoods(points, mu, var)

    # sum the densities to get mixture density
    Z = torch.sum(P, dim=0).data.numpy().reshape([N, N])
    
    return X, Y, Z

def plot_density(X, Y, Z, i=0):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.inferno)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.inferno)

    # adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()

def EM_routing(R, V):
    eps = 1e-6
    iteration = 10
    
    for i in range(iteration):
        # M-step
        #R = (R * a_).unsqueeze(-1)
        R = R.unsqueeze(-1)
        sum_R = R.sum(1)
        mu = ((R * V).sum(1) / sum_R).unsqueeze(1)
        sigma_square = ((R * (V - mu) ** 2).sum(1) / sum_R).unsqueeze(1)

        #cost = (self.beta_v.view(kaj) + torch.log(sigma_square.sqrt().view(b_C_w)+self.eps)) * sum_R.view(b_C_w)
        #a = torch.sigmoid(lambda_ * (self.beta_a.view(aag) - cost.sum(-1)))
        #a = a.view(self.b, -1)

        # E-step
        if i != iteration - 1:
            normal = Normal(mu, sigma_square.sqrt())
            p = torch.exp(normal.log_prob(V+eps))
            ap = p.prod(-1)

            plot_density(*get_density(mu.squeeze(1), sigma_square.squeeze(1), pi, N=100, X_range=(-2, 12), Y_range=(-2, 12)), i=i)
            #XYZ = torch.stack([V[:,:,0],V[:,:,1],ap], dim=-1)
            #plotXYZ(XYZ)
            
            #ap = a__.unsqueeze(1) * p.sum(-1)
            R = Variable(ap / ap.sum(0, keepdim=True), requires_grad=False) + eps
            
            #R = Variable(ap / ap.sum(-1, keepdim=True), requires_grad=False) + eps

    return mu


# training loop
k = 3
d = 2
nb_iters = 1000

data = Variable(X)

# tile the design matrix `K` times on the batch dimension 
X = data.unsqueeze(0).repeat(k, 1, 1)


"""
Randomly initialize the parameters for `k` gaussians.

data: design matrix (examples, features)
k: number of gaussians
var: initial variance
"""

# choose k points from data to initialize means
var = 1
m = data.size(0)
idxs = Variable(torch.from_numpy(
    np.random.choice(m, k, replace=False)))
mu = data[idxs].unsqueeze(1)

# uniform sampling for means and variances
var = Variable(torch.Tensor(k, d).fill_(var)).unsqueeze(1)

# equal priors
pi = Variable(torch.Tensor(k).fill_(1)) / k





print(mu.size())

prev_cost = float('inf')
thresh = 1e-4
eps = 1e-6
for i in tqdm(range(nb_iters)):

    # E-STEP

    # get the likelihoods p(x|z) under the parameters
    #p = get_k_likelihoods(data, mu, var)

    normal = Normal(mu, var.sqrt())
    P = torch.exp(normal.log_prob(X))
    P = P.prod(-1)
    
    #XYZ = torch.stack([X[:,:,0],X[:,:,1],P], dim=-1)
    #plotXYZ(XYZ)

    # plot!
    #x,y,z = getXYZ(data, P)
    #plot_density(x,y,z)
    plot_density(*get_density(mu.squeeze(1), var.squeeze(1), pi, N=100, X_range=(-2, 12), Y_range=(-2, 12)), i=i)
    
    # compute the "responsibilities" p(z|x)
    # P: the relative likelihood of each data point under each gaussian (K, examples)
    P_sum = torch.sum(P, dim=0, keepdim=True)

    # p(z|x): (K, examples)
    gamma = P / (P_sum+eps)


    EM_routing(gamma, X)
    
    # compute the cost

    """
    Get the log-likelihood of the data points under the given distribution.
    P: likelihoods / densities under the distributions. (k, examples)
    pi: priors (K)
    """
    
    # get weight probability of each point under each k
    sum_over_k = torch.sum(pi.unsqueeze(1) * P, dim=0)
    
    # take log probability over each example `m`
    sum_over_m = torch.sum(torch.log(sum_over_k + eps))
    
    # divide by number of training examples
    cost = -sum_over_m / P.size(1)

    # check for convergence
    diff = prev_cost - cost
    if torch.abs(diff).data[0] < thresh:
        break
    prev_cost = cost



    # M-STEP

    # re-compute parameters

    # compute `N_k` the proxy "number of points" assigned to each distribution.
    N_k = torch.sum(gamma, dim=1) + eps # (K)
    N_k = N_k.view(k, 1, 1)


    # get the means by taking the weighted combination of points
    mu = (gamma.unsqueeze(2) * X).sum(1, keepdim=True)
    #mu = gamma.unsqueeze(1) @ X # (K, 1, features)
    mu = mu / N_k
    
    # compute the diagonal covar. matrix, by taking a weighted combination of
    # the each point's square distance from the mean
    A = X - mu
    var = (gamma.unsqueeze(2) * (A ** 2)).sum(1, keepdim=True)
    #var = gamma.unsqueeze(1) @ (A ** 2) # (K, 1, features)
    var = var / N_k

    # recompute the mixing probabilities
    pi = N_k / N_k.sum()
