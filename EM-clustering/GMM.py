import torch
import numpy as np
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.distributions import Normal
import torch.nn as nn

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

def plotXYZ(XYZ):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    for i in range(XYZ.shape[0]):
        for j in range(XYZ.shape[1]):
            xs, ys, zs = XYZ[i,j,:]
            ax.scatter(xs, ys, zs, c='r', marker='o')
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()

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
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.inferno)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.inferno)

    # adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()


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

clusterNoise = sample(
    torch.Tensor([10, 10]),
    torch.Tensor([.5, .5]), nb_samples=100
)

cluster1 = torch.cat([cluster1, clusterNoise], dim=0)
cluster2 = torch.cat([cluster2, clusterNoise], dim=0)
cluster3 = torch.cat([cluster3, clusterNoise], dim=0)

X = torch.stack([cluster1, cluster2, cluster3], dim=1)
plot_2d_sample(X.view(-1,2))


class myModel(nn.Module):

    def __init__(self):
        super(myModel, self).__init__()    
        
        self.b = 1
        self.Bkk = 600
        self.C = 3
        self.Cww = 3
        self.hh = 2
        self.iteration = 3
        self.eps = 1e-10
        self.ln_2pi = torch.FloatTensor(1).fill_(math.log(2*math.pi))
        self.beta_v = nn.Parameter(torch.randn(self.C).view(1,self.C,1,1))
        self.beta_a = nn.Parameter(torch.randn(self.C).view(1,self.C,1))

        #self.beta_v = nn.Parameter(torch.zeros(self.C).view(1,self.C,1,1))
        #self.beta_v = nn.Parameter((torch.FloatTensor(self.C) + 0.5 + 0.5*self.ln_2pi).view(1,self.C,1,1))
        #self.beta_a = nn.Parameter(torch.zeros(self.C).view(1,self.C,1))
        
    def EM_routing_new(self, lambda_, a_, V):
        # routing coefficient
        R = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False) / self.C
    
        for i in range(self.iteration):
            # M-step
            R = (R * a_).unsqueeze(-1)
            sum_R = R.sum(1)
            mu = ((R * V).sum(1) / sum_R).unsqueeze(1)
            V_minus_mu_sqr = (V - mu) ** 2
            sigma_square = ((R * V_minus_mu_sqr).sum(1) / sum_R).unsqueeze(1)
    
            debug = False
            if debug:
                plot_density(*get_density(mu.squeeze(), sigma_square.squeeze(), N=100, X_range=(-2, 12), Y_range=(-2, 12)))
    
            """
            beta_v: Bias for scaling
            beta_a: Bias for offsetting
            """
            log_sigma = torch.log(sigma_square.sqrt()+self.eps)
            #cost = (self.beta_v + log_sigma.view(self.b,self.C,-1,self.hh) + 0.5 + 0.5*self.ln_2pi) * sum_R.view(self.b, self.C,-1,1)
            cost = (self.beta_v + log_sigma.view(self.b,self.C,-1,self.hh)) * sum_R.view(self.b, self.C,-1,1)
            #a = torch.softmax(lambda_ * (self.beta_a - cost.sum(-1)), dim=1)
            a = torch.sigmoid(lambda_ * (self.beta_a - cost.sum(-1)))
            a = a.view(self.b, self.Cww)

            # E-step
            if i != self.iteration - 1:
                ln_p_j_h = -V_minus_mu_sqr / (2 * sigma_square) - log_sigma - 0.5*self.ln_2pi
                p = torch.exp(ln_p_j_h)
                ap = a[:,None,:] * p.sum(-1)
                #ap = R.data.squeeze(-1) * p.sum(-1)
                #R = ap / (torch.sum(ap, 2, keepdim=True) + self.eps) + self.eps
                R = Variable(ap / (torch.sum(ap, 2, keepdim=True) + self.eps) + self.eps, requires_grad=False)
    
        return a, mu
    
    def forward(self, x, a, lambda_):
        a, p = self.EM_routing_new(lambda_, a, x)
        return p, a
        

    
model = myModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=True)
loss = nn.MSELoss(reduction='sum')

votes = X.view(1,model.Bkk,model.Cww,model.hh)
votes = Variable(votes)
labels = torch.tensor([[0.4, 0.5, 0.7]])
#labels = torch.tensor([[[[2.4378, 2.5000],
#                         [9.0173, 8.9117],
#                         [8.1716, 1.7770]]]])

labels = Variable(labels)

a = Variable(torch.ones([model.b, model.Bkk, model.Cww]), requires_grad=False)

lambda_ = 0.00001
delta = 0.

for epoch in range(20000):

    optimizer.zero_grad()
    #a = a.detach()
    votes_, a_ = model(votes, a, lambda_)

    main_loss = loss(a_.view(-1), labels.view(-1))
    main_loss.backward()
    optimizer.step()

    print("epoch",epoch,"loss",main_loss, a_)
    
    #if epoch < 10000:
    #    delta = 0.0000001
    #else:
    #    delta = 0.000005
        #delta += 0.0000000025

    #lambda_ += delta

    lambda_ += 0.00005

#EM_routing(self, 1.0, votes)
