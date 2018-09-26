'''
Created on Jun 30, 2018

@author: jens
'''
# Code in file nn/two_layer_net_nn.py
import torch
from torch.optim import lr_scheduler
import pyrr
import random
import math
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cpu')


class SpikeNet(nn.Module):
    def __init__(self, input, output):
        super(SpikeNet, self).__init__()
        self.linear = nn.Linear(input, output)
        self.y_remain = 0
        
    def forward(self, x):
        y_orig = self.linear(x)
        #return F.relu(y_orig)
        threshold = y_orig.data.max() / 2
        y_out = 0
        self.y_remain = 0

        for i in range(10):
            y = y_orig.data + self.y_remain * 0.5 # decay
            #threshold = y.max() / 2.0
            y_spike = y.clone()
            y_spike[y_spike<threshold] = 0
            #y_spike = F.relu(y_spike) + threshold
            y_out += y_spike
            self.y_remain = y - y_spike
        
        y_out[y_out<0] = 0
        y_out[y_out>0] = 1

        y_orig *= y_out
        
        return y_orig

    """
    def forward(self, x, initial=True):
        y_orig = self.linear(x)
        if initial:
            self.threshold = y_orig.data.max() / 3
            y = y_orig.data
        else:
            y = y_orig.data + self.y_remain * 0.75 # decay
            
        y_spike = y.clone()
        y_spike[y_spike<threshold] = 0

        self.y_remain = y - y_spike
        
        y_out[y_out<0] = 0
        y_out[y_out>0] = 1

        y_orig *= y_out
        
        return y_orig
    """

model = nn.Sequential(
    SpikeNet(1,10),
    SpikeNet(10,200),
    nn.Linear(200, 1),
    nn.Tanh()
)

loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True) #, eps=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)


while True:
    sum_loss = 0
    for t in range(5000):
        
        x = random.random()*2*math.pi
        y = math.sin(x)
        
        x, y = Variable(torch.tensor([x])), Variable(torch.tensor([y]))
        
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        #print(t, loss.item())
      
        model.zero_grad()
        loss.backward()
    
        optimizer.step()
        sum_loss += loss.data
        
    #scheduler.step(sum_loss)
    print(sum_loss)
