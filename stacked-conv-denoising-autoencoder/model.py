import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F



class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.

    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()

        # jeo stuff
        self.K = 2
        self.st = stride
        #self.W = nn.Parameter(torch.randn(output_size, input_size, self.K, self.K))

        
        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=0), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward_pass(x_noisy)
        
        
        # test
        #x = torch.tensor(torch.arange(16)).float().view(1,1,4,4)
        #W = torch.tensor([0,1,1,0]).float().view(1,2,2,1)

        width_in = x_noisy.size(2)
        w = int((width_in - self.K) / self.st + 1)
        poses = torch.stack([x_noisy[:,:,self.st * i:self.st * i + self.K, self.st * j:self.st * j + self.K] for i in range(w) for j in range(w)], dim=-1)
        poses = poses[:,None,:,:,:,:,None,None]
        W_hat = self.forward_pass[0].weight[None,:,:,:,:,None,None,None]
        #poses = poses[:,:,:,:,None,None]
        #W_hat = W[:,:,None,None,None]
        votes = W_hat @ poses
        votes = votes.view(x_noisy.shape[0],W_hat.shape[1],-1,w,w).sum(2)
        votes = votes + self.forward_pass[0].bias[None,:,None,None]
        _y = F.relu(votes)

        # test        
        #print (votes.view(-1,w,w).sum(0))
        
        

        if self.training:
            x_reconstruct = self.backward_pass(y)
            
            width = self.K*self.st + x.shape[3]
            pad = int((width - x.shape[3]) / 2)
            
            zero = torch.zeros(_y.shape[0], _y.shape[1], _y.shape[3], 1).cuda()
            padding = torch.zeros(_y.shape[0], _y.shape[1], _y.shape[3], pad).cuda()
            y_expanded = torch.cat([torch.cat([_y[:,:,:,i:i+1], zero], dim=3) for i in range(_y.shape[3])], dim=3)
            y_expanded = torch.cat([padding,y_expanded,padding], dim=3)
            zero = torch.zeros(_y.shape[0], _y.shape[1], 1, width).cuda()
            padding = torch.zeros(_y.shape[0], _y.shape[1], pad, width).cuda()
            y_expanded = torch.cat([torch.cat([y_expanded[:,:,i:i+1,:], zero], dim=2) for i in range(_y.shape[2])], dim=2)
            y_expanded = torch.cat([padding,y_expanded,padding], dim=2)
            
            
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = CDAutoEncoder(3, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct
