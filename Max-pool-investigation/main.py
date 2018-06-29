'''
Created on Jun 28, 2018

@author: jens
'''

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=3, padding=1)
        #nn.ReLU(True),
        #nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8,
                               kernel_size=3, stride=2, padding=1)
        #nn.ReLU(True),
        #nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2


        self.convtrans1 = nn.ConvTranspose2d(in_channels=8, out_channels=16,
                                kernel_size=3, stride=2, padding=1)  # b, 16, 5, 5
        #nn.ReLU(True),
        self.convtrans2 = nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                kernel_size=3, stride=3, padding=1)  # b, 8, 15, 15
        #nn.ReLU(True),
        #self.convtrans3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #nn.Tanh()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, id1 = F.max_pool2d(x,2, 2, return_indices=True)
        x = F.relu(self.conv2(x))
        x, id2 = F.max_pool2d(x,2, 1, return_indices=True)

        x = F.max_unpool2d(x, id2, 2, 1)
        x = self.convtrans1(x)
        x = F.relu(x)
        x = F.max_unpool2d(x, id1, 2, 2)
        x = F.tanh(self.convtrans2(x))
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
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
