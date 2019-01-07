'''
Created on Aug 22, 2018

@author: jens
'''

import torch.nn as nn

class PoseToQuatNet(nn.Module):
    def __init__(self):
        super(PoseToQuatNet, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 7),
            nn.Tanh()
            #nn.Sigmoid()
        )

    def forward(self, x):
        y = self.decoder(x)
        return y
