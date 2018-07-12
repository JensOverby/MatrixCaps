import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import pyrr
import random
import math

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        self.inputLayer = nn.Linear(16, 128)
        self.inputLayer.weight.data *= 50.0
        self.hiddenLayer = nn.Linear(128, 2048)
        self.hiddenLayer.weight.data *= 0.1
        self.outputLayer = nn.Linear(2048, 7)

    def forward(self, x):
        x = self.inputLayer(x)
        x = F.relu(x)
        x = self.hiddenLayer(x)
        x = F.relu(x)
        x = self.outputLayer(x)
        x = F.tanh(x)
        return x

#        self.reconstruction_loss = nn.MSELoss(size_average=False)
#        reconstruction_loss = self.reconstruction_loss(reconstructions, images)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt

    model = TestNet()
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    #model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters(), lr=1e-4)

    test_loss = nn.MSELoss(size_average=False)


    while True:
        mat_list = []
        lab_list = []
        
        for i in range(64):
            rot = pyrr.Matrix44.from_x_rotation(random.random()*2*math.pi)
            rot *= pyrr.Matrix44.from_y_rotation(random.random()*2*math.pi)
            trans = pyrr.Matrix44.identity(float)
            trans[3,0] = random.random()*1.0 - 0.5
            trans[3,1] = random.random()*0.7 - 0.35
            trans[3,2] = random.random()*0.4 - 0.2
    
            q = pyrr.Quaternion.from_matrix(rot)
            
            mat = trans*rot
    
            m = pyrr.matrix44.create_from_quaternion(q)
            
            labels = np.asarray([trans[3,0],trans[3,1],trans[3,2],q[0],q[1],q[2],q[3]])
            
            mat_list.append(mat)
            lab_list.append(labels)
            
            
        
        
        

        data = Variable( torch.tensor(mat_list).float() ) #.cuda()
        labels = Variable( torch.tensor(lab_list).float() ) #.cuda()

        output = model(data.view(data.shape[0], -1))

        loss = test_loss(output, labels)
        
        loss.backward()

        print ('loss=',loss)
