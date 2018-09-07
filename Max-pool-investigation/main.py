'''
Created on Jun 28, 2018

@author: jens
'''

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 1 #128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class MyPool(nn.Module):
    def __init__(self, intermediate=False):
        super(MyPool, self).__init__()
        if intermediate:
            first = nn.Linear(6, 4)
        else:
            first = nn.Linear(4, 4)
        self.net = nn.Sequential(
            first,
            nn.ReLU(inplace=True),
            nn.Linear(4, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 4),
            nn.ReLU(inplace=True) )
        
    def forward(self, x, i=None):
        if i != None:
            x = x.cat(i)
        x = self.net(x)
        return x[:2], x[2:]

class LearnPool(nn.Module):
    def __init__(self):
        super(LearnPool, self).__init__()
        self.W = nn.Parameter(torch.randn(2*2))
        
    def forward(self, x):
        d = x.shape[-1]
        x, id = F.max_pool2d(x,2, 2, return_indices=True)
        id1 = torch.fmod(id, d*2)
        route = torch.fmod(id1, 2) + 2*(id1/d)
        route = route.float()
        route[route==0] = self.W[0]
        route[route==1] = self.W[1]
        route[route==2] = self.W[2]
        route[route==3] = self.W[3]
        x = route * x
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.pool1 = MyPool()
        self.conv2 = nn.Conv2d(32, 1, 3, stride=2, padding=1)
        self.pool2 = MyPool(intermediate=True)
        #self.conv3 = nn.Conv2d(16, 8, 4, stride=1)

        """        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2),  # b, 16, 13, 13
            nn.ReLU(True),
            LearnPool(),
            nn.Conv2d(16, 16, 3, stride=2),  # b, 16, 6, 6
            nn.ReLU(True),
            nn.Conv2d(16, 8, 4, stride=1),  # b, 8, 3, 3
            nn.ReLU(True)
            
            #nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            #nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            #nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            #nn.ReLU(True),
            #nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        """
        
        self.decoder = nn.Sequential(
            nn.Linear(4, 256),   # 16 - 1024 - 10240 - 10000
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x,l = self.pool1(x)
        x = F.relu(self.conv2(x))
        l = F.relu(self.conv2(l))
        x,l = self.pool2(x,l)
        x = F.tanh(x)
        #x = F.relu(self.conv3(x))
        
        #x = self.encoder(x)
        x = self.decoder(x.view(-1))
        return x

import numpy as np
a,b,c,d,e,f = 1/3,2/4,3/5,4/6,5/7,6/8

normal = np.array([
    1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,f,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,e,f,1,1,1,1,1,1,f,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,d,e,f,1,1,1,1,1,e,f,1,1,1,1,1,1,f,1,1,1,1,1,1,1,0,f,e,d,c,b,a,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,c,d,e,f,1,1,1,1,d,e,f,1,1,1,1,1,e,f,1,1,1,1,1,1,f,1,f,e,d,c,b,a,0,f,e,d,c,b,a,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,b,c,d,e,f,1,1,1,c,d,e,f,1,1,1,1,d,e,f,1,1,1,1,1,e,f,1,f,e,d,c,b,a,1,f,e,d,c,b,a,0,f,e,d,c,b,a,0,0,0,0,0,0,0,0,
    1,a,b,c,d,e,f,1,1,b,c,d,e,f,1,1,1,c,d,e,f,1,1,1,1,d,e,f,1,f,e,d,c,b,1,1,f,e,d,c,b,a,1,f,e,d,c,b,a,0,f,e,d,c,b,a,0,
    0,a,b,c,d,e,f,1,a,b,c,d,e,f,1,1,b,c,d,e,f,1,1,1,c,d,e,f,1,f,e,d,c,1,1,1,f,e,d,c,b,1,1,f,e,d,c,b,a,1,f,e,d,c,b,a,0,
    0,a,b,c,d,e,f,0,a,b,c,d,e,f,1,a,b,c,d,e,f,1,1,b,c,d,e,f,1,f,e,d,1,1,1,1,f,e,d,c,1,1,1,f,e,d,c,b,1,1,f,e,d,c,b,a,1,
    0,0,0,0,0,0,0,0,a,b,c,d,e,f,0,a,b,c,d,e,f,1,a,b,c,d,e,f,1,f,e,1,1,1,1,1,f,e,d,1,1,1,1,f,e,d,c,1,1,1,f,e,d,c,b,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,a,b,c,d,e,f,0,a,b,c,d,e,f,1,f,1,1,1,1,1,1,f,e,1,1,1,1,1,f,e,d,1,1,1,1,f,e,d,c,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,a,b,c,d,e,f,0,1,1,1,1,1,1,1,f,1,1,1,1,1,1,f,e,1,1,1,1,1,f,e,d,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,f,1,1,1,1,1,1,f,e,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,f,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1
]).reshape(15,57) #.transpose(1,0)

def n(v):
    i = v*29 + 28
    return normal[i,:]

vec = []
for _ in range(12):
    vec.append( np.array(np.random.rand(16)*2-1) )
vec = np.concatenate(vec).reshape(16,-1)

kaj = (vec*29+28).astype(int)

aage = np.stack([normal[:,kaj[i,j]] for i in range(kaj.shape[0]) for j in range(kaj.shape[1])]).reshape(kaj.shape[0],kaj.shape[1],-1)


for _ in range(5):
    dist = aage.sum(1)
    routing = []
    for i in range(aage.shape[1]):
        dist_part = aage[:,i,:] * dist
        routing.append(dist_part.sum())
    routing = np.concatenate([routing])
    routing = routing / np.amax(routing)
    print('routing:', routing)
    aage = routing.reshape(1,-1,1) * routing.reshape(1,-1,1) * aage

#            activations_ = torch.stack([activations[:, :, self.stride * i:self.stride * i + self.K,
#                                 self.stride * j:self.stride * j + self.K] for i in range(w) for j in range(w)],
#                                dim=-1)  # b,B,K,K,w*w
#            activations_ = activations_.view(self.b, self.Bkk, 1, -1).repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)




model = autoencoder().cpu() #cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cpu() #cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img.view(-1))
        print('loss=',loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
