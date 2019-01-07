import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def calc_out(input, kernel=1, stride=1, padding=0, dilation=0):
    return (input + 2*padding - dilation*(kernel-1) - 1) / stride

def calc_same_padding(input_, kernel=1, stride=1, dilation=1):
    return (dilation*(kernel-1) + 1) // 2, input_ // stride

#def calc_same_padding(size, kernel_size=1, stride=1, dilation=0):
#    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) //2
#    return padding

class ConvCapsuleLayer(torch.nn.Module):
    def __init__(self, output_dim, input_atoms, output_atoms, num_routing, stride, kernel_size, padding=0, type_='Conv2d', use_cuda=True):
        super(ConvCapsuleLayer, self).__init__()

        """ scaling ~ stride """

        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.use_cuda = use_cuda

        if type_ == 'Conv2d':
            self.conv = nn.Conv2d(in_channels=input_atoms,
                                           out_channels=output_dim * output_atoms,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        elif type_ == 'Conv3d':
            self.conv = nn.Conv3d(in_channels=input_atoms,
                                           out_channels=output_dim * output_atoms,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        elif type_ == 'ConvTranspose2d':
            self.conv = nn.ConvTranspose2d(in_channels=input_atoms,
                                           out_channels=output_dim * output_atoms,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        elif type_ == 'ConvTranspose3d':
            self.conv = nn.ConvTranspose3d(in_channels=input_atoms,
                                           out_channels=output_dim * output_atoms,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=False)
        else:
            raise NotImplementedError('Convolutional type not recognized. Must be: Conv2d, Conv3d, ConvTranspose2d, or ConvTranspose3d"')


        nn.init.normal(self.conv.weight.data, mean=0,std=0.1)

        self.bias = nn.Parameter(torch.Tensor(output_dim, output_atoms))
        nn.init.constant(self.bias.data, val=0.1)

    def forward(self, x):
        x_shape = x.size()
        x = x.view(x_shape[0]*x_shape[1], x_shape[2], x_shape[3], x_shape[4])
        x = self.conv(x)

        votes = x.view(x_shape[0], x_shape[1], self.output_dim, self.output_atoms, x.size(2), x.size(3))
        
        biases_replicated = self.bias[...,None,None].repeat([1,1,x.size(2),x.size(3)])
        logit_shape = list(votes.size())
        logit_shape.pop(3)

        return _routing(votes=votes, biases=biases_replicated, logit_shape=logit_shape, num_routing=self.num_routing, use_cuda=self.use_cuda)

class Reconstruction(torch.nn.Module):
    def __init__(self, num_classes, num_atoms, layer_sizes, num_pixels):
        super(Reconstruction, self).__init__()

        self.num_atoms = num_atoms

        first_layer_size, second_layer_size = layer_sizes

        self.dense0 = nn.Linear(num_atoms*num_classes, first_layer_size)
        nn.init.normal(self.dense0.weight.data, mean=0,std=0.1)
        nn.init.constant(self.dense0.bias.data, val=0.0)
        self.dense1 = nn.Linear(first_layer_size, second_layer_size)
        nn.init.normal(self.dense1.weight.data, mean=0,std=0.1)
        nn.init.constant(self.dense1.bias.data, val=0.0)
        self.dense2 = nn.Linear(second_layer_size, num_pixels)
        nn.init.normal(self.dense2.weight.data, mean=0,std=0.1)
        nn.init.constant(self.dense2.bias.data, val=0.0)

    def forward(self, capsule_embedding, capsule_mask):
        atom_mask = capsule_mask.clone().unsqueeze(-1).expand(capsule_embedding.size())
        filtered_embedding = capsule_embedding * atom_mask
        filtered_embedding = filtered_embedding.view(filtered_embedding.size(0), -1)

        x = F.relu(self.dense0(filtered_embedding))
        x = F.relu(self.dense1(x))
        x = F.sigmoid(self.dense2(x))
        return x

def _routing(votes, biases, logit_shape, num_routing, use_cuda=False):
    votes_trans = votes.permute(3, 0, 1, 2, 4, 5)
    votes_trans_stopped = votes_trans.clone().detach()

    logits = Variable(torch.zeros(logit_shape), requires_grad=False)

    if use_cuda:
        logits = logits.cuda()

    for i in range(num_routing):
        route = F.softmax(logits, 2)

        if i == num_routing - 1:
            preactivate_unrolled = route * votes_trans
            preact_trans = preactivate_unrolled.permute(1, 2, 3, 0, 4, 5)
            preactivate = torch.sum(preact_trans, dim=1) + biases
            activation = _squash(preactivate)
        else:
            preactivate_unrolled = route * votes_trans_stopped
            preact_trans = preactivate_unrolled.permute(1, 2, 3, 0, 4, 5)
            preactivate = torch.sum(preact_trans, dim=1) + biases
            activation = _squash(preactivate)

            act_3d = activation.unsqueeze_(1)
            distances = torch.sum(votes * act_3d, dim=3)
            logits = logits + distances
            
    return activation

def _squash(input_tensor):
    norm = torch.norm(input_tensor, p=2, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
