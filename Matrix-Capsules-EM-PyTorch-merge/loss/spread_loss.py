import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

import torch.nn.functional as F
from torch.autograd import Variable

class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10, use_recon=True):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class
        self.use_recon = use_recon
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, x, target, r, images, recon):
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min)*r

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        if self.use_recon:
            recon_loss = self.reconstruction_loss(recon, images)
            loss += 0.00005 * recon_loss

        return loss

class CapsuleLoss(_Loss):
    def __init__(self, args):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.args = args

    @staticmethod
    def spread_loss(x, target, r):  # x:b,10 target:b
        loss = F.multi_margin_loss(x, target, p=2, margin=r)
        return loss

    @staticmethod
    def cross_entropy_loss(x, target, r):
        loss = F.cross_entropy(x, target)
        return loss

    def margin_loss(self, x, target, r):
        left = F.relu(0.9 - x, inplace=True) ** 2
        right = F.relu(x - 0.1, inplace=True) ** 2

        target = Variable(torch.eye(self.args.num_classes).cuda()).index_select(dim=0, index=target)

        margin_loss = target * left + 0.5 * (1. - target) * right
        margin_loss = margin_loss.sum()
        return margin_loss * 1/x.size(0)

    #def forward(self, images, output, labels, m, recon):
    def forward(self, x, target, r, images, recon):
        main_loss = getattr(self, self.args.loss)(x, target, r)

        if self.args.use_recon:
            recon_loss = self.reconstruction_loss(recon, images)
            main_loss += 0.0005 * recon_loss

        return main_loss/10.0
