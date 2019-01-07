'''
Created on Oct 7, 2018

@author: jens
'''

import torch
from torch.autograd import Variable
import torch.nn as nn
import model.capsules as caps
import model.tools as tools

def sample(mu, var, nb_samples=500):
    """
    Return a tensor of (nb_samples, features), sampled
    from the parameterized gaussian.
    
    :param mu: torch.Tensor of the means
    :param var: torch.Tensor of variances (NOTE: zero covars.)
    """
    out = []
    for _ in range(nb_samples):
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

clusterNoise = sample(
    torch.Tensor([10, 10]),
    torch.Tensor([.5, .5]), nb_samples=100
)

cluster1 = torch.cat([cluster1, clusterNoise], dim=0)
cluster2 = torch.cat([cluster2, clusterNoise], dim=0)
cluster3 = torch.cat([cluster3, clusterNoise], dim=0)

X = torch.stack([cluster1, cluster2, cluster3], dim=1)
tools.plot_2d_sample(X.view(-1,2))


model = caps.ConvCaps()
model.b = 1
model.Bkk = 600
model.C = 3
model.Cww = 3
model.hh = 2
model.iteration = 3
model.beta_v = nn.Parameter((torch.FloatTensor(model.C) + 0.5 + 0.5*model.ln_2pi).view(1,model.C,1,1))
model.beta_a = nn.Parameter(torch.zeros(model.C).view(1,model.C,1))
model.testing = True

        


#model = myModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=True)
loss = nn.MSELoss(reduction='sum')

votes = X.view(1,model.Bkk,model.Cww,model.hh)
votes = Variable(votes)
#labels = torch.tensor([[[[2.4378, 2.5000],
#                         [9.0173, 8.9117],
#                         [8.1716, 1.7770]]]])
labels = torch.tensor([[0.4, 0.5, 0.7]])

labels = Variable(labels)

a = Variable(torch.ones([model.b, model.Bkk, model.Cww]), requires_grad=False)

lambda_ = 0.00001

for epoch in range(20000):

    optimizer.zero_grad()

    a_, votes_ = model.EM_routing(lambda_, a, votes)

    main_loss = loss(a_.view(-1), labels.view(-1))
    main_loss.backward()
    optimizer.step()

    print("epoch",epoch,"loss",main_loss, a_)
    
    debug = False
    if debug:
        tools.plot_density(*tools.get_density(model.mu.squeeze(), model.sigma_square.squeeze(), N=100, X_range=(-2, 12), Y_range=(-2, 12)))

    lambda_ += 0.00001

    """
    mu_expanded = model.mu.expand(votes.shape)
    sigma_expanded = model.sigma_square.sqrt().expand(1,600,3,2)
    votes = torch.normal(mu_expanded, sigma_expanded)
    V_minus_mu_sqr = (votes - model.mu) ** 2

    ln_p_j_h = -V_minus_mu_sqr / (2 * self.sigma_square) - log_sigma - 0.5*self.ln_2pi
    p = torch.exp(ln_p_j_h)
    ap = a[:,None,:] * p.sum(-1)
    R = Variable(ap / (torch.sum(ap, 2, keepdim=True) + self.eps) + self.eps, requires_grad=False) # detaches from graph
    R_orig = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False) / self.C
    R = (R * a_).unsqueeze(-1)
    """
    