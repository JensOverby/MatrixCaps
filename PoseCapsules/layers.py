import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

def calc_out(input, kernel=1, stride=1, padding=0, dilation=1):
    return int((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1

def calc_same_padding(input_, kernel=1, stride=1, dilation=1, transposed=False):
    if transposed:
        return (dilation*(kernel-1) + 1) // 2 - 1, input_ // (1./stride)
    else:
        return (dilation*(kernel-1) + 1) // 2, input_ // stride

def make_decoder(dec_list):
    model = nn.Sequential()
    for i in dec_list:
        model.add_module(*i)
    return model

def make_decoder_list(layer_sizes, out_activation, mean=0.0, std=0.1, bias=0.0):
    last_layer_size = layer_sizes[0]
    layer_sizes.pop(0)
    sz = len(layer_sizes)
    #layer_list = OrderedDict()
    model = [] #nn.Sequential()
    for i, layer_size in enumerate(layer_sizes, 1):
        name = 'layer' + str(i)
        layer = nn.Linear(last_layer_size, layer_size)
        nn.init.normal_(layer.weight.data, mean=mean, std=std)
        nn.init.constant_(layer.bias.data, val=bias)
        model.append((name, layer))
        #layer_list[name] = layer
        last_layer_size = layer_size
        if i < sz:
            name = 'relu' + str(i)
            model.append((name, nn.ReLU(inplace=True)))
    if out_activation=='sigmoid':
        model.append(('sigm', nn.Sigmoid()))
    elif out_activation=='tanh':
        model.append(('tanh', nn.Tanh()))
    elif out_activation=='relu':
        model.append(('relu', nn.ReLU(inplace=True)))
    else:
        model.append(('softplus', nn.Softplus()))
    return model

class PrimToCaps(nn.Module):
    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2).contiguous()                   # batch_size, input_dim, dim_x, dim_y, input_atoms
        x = x.view(x.size(0), -1, x.size(-1), 1, 1)                 # batch_size, input_dim*dim_x*dim_y, input_atoms, 1, 1
        """
        shp = x.size()
        act = x.norm(dim=2, keepdim=True)
        act_sorted = act.view(shp[0],-1).sort(-1, descending=True)
        act_sorted = act_sorted[1].repeat(1,shp[2])
        x = torch.gather(x.view(shp[0],-1), 1, act_sorted).view(shp)
        """
        return x

class ConvToPrim(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)

class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MyConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) #.cuda(self.device)
        nn.init.normal_(self.weight.data, mean=0,std=0.1)
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x_orig):
        """
        shp = list(x_orig.size())
        shp[0] = self.weight.shape[1]
        shp.insert(0, -1)
        x = x_orig.view(shp).permute(0,2,1,3,4).contiguous()
        shp.pop(2)
        x = x.view(shp)
        """
        x_sh = x.size() # batch_size*input_dim, input_atoms, dim_x, dim_y
        w = int((x_sh[2] - self.kernel_size) / self.stride) + 1
        pose_list = []
        for i in range(w):
            for j in range(w):
                tile = x[:, :, self.stride * i:self.stride * i + self.kernel_size, self.stride * j:self.stride * j + self.kernel_size]
                tile = self.weight[None,...] * tile[:,None,...]
                tile = tile.view(x_sh[0], self.out_channels,-1).sum(-1)
                pose_list.append( tile )
        poses = torch.stack(pose_list, dim=-1)
        poses = poses.view(x_sh[0], self.out_channels, w, w)
        return poses

class CapsuleLayer(nn.Module):
    
    """
    output_dim:    number of classes
    """
    
    def __init__(self, output_dim, output_atoms, num_routing, voting, device = torch.device('cuda')):
        super(CapsuleLayer, self).__init__()

        self.not_initialized = True
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.voting = voting
        self.device = device
        self.bias = nn.Parameter(torch.Tensor(self.output_dim, self.output_atoms, 1, 1))
        nn.init.constant_(self.bias.data, val=0.1)

    def init(self, input_dim, input_atoms):
        if self.voting['type'] == 'standard' or self.voting['type'] == 'standard_sort':
            self.weights = nn.Parameter(torch.Tensor(input_dim, input_atoms, self.output_dim * self.output_atoms, 1, 1)).cuda(self.device)
        elif self.voting['type'] == 'Conv2d':
            self.conv = nn.Conv2d(in_channels=input_atoms,
                                           out_channels=self.output_dim * self.output_atoms,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'],
                                           padding=self.voting['padding'],
                                           bias=False)
            #self.batchnorm = nn.BatchNorm2d(num_features=self.output_dim * self.output_atoms, eps=0.001, momentum=0.1, affine=True)
        elif self.voting['type'] == 'Conv3d':
            self.conv = nn.Conv3d(in_channels=input_atoms,
                                           out_channels=self.output_dim * self.output_atoms,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'],
                                           padding=self.voting['padding'],
                                           bias=False)
        elif self.voting['type'] == 'ConvTranspose2d':
            self.conv = nn.ConvTranspose2d(in_channels=input_atoms,
                                           out_channels=self.output_dim * self.output_atoms,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'],
                                           padding=self.voting['padding'],
                                           bias=False)
        elif self.voting['type'] == 'ConvTranspose3d':
            self.conv = nn.ConvTranspose3d(in_channels=input_atoms,
                                           out_channels=self.output_dim * self.output_atoms,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'],
                                           padding=self.voting['padding'],
                                           bias=False)
        elif self.voting['type'] == 'experimental':
            self.conv = MyConv(in_channels=input_atoms,
                                           out_channels=self.output_dim * self.output_atoms,
                                           kernel_size=self.voting['kernel_size'],
                                           stride=self.voting['stride'])
        else:
            raise NotImplementedError('Convolutional type not recognized. Must be: Conv2d, Conv3d, ConvTranspose2d, or ConvTranspose3d"')

        if self.voting['type'] == 'standard':
            nn.init.normal_(self.weights.data, mean=0,std=0.1)      #input_dim, input_atoms, output_dim*output_atoms
            self.type = 1
        elif self.voting['type'] == 'standard_sort':
            nn.init.normal_(self.weights.data, mean=0,std=0.1)      #input_dim, input_atoms, output_dim*output_atoms
            self.type = 2
        else:
            nn.init.normal_(self.conv.weight.data, mean=0,std=0.1)
            self.conv.cuda(self.device)
            #self.batchnorm.cuda(self.device)
            self.type = 0
        self.not_initialized = False


    def forward(self, x):
        x_sh = x.size()                                             # batch_size, input_dim, input_atoms, dim_x, dim_y
        if self.not_initialized:
            self.init(x_sh[1], x_sh[2])
            
        if self.type == 1:
            x.unsqueeze_(3)                                         # batch_size, input_dim, input_atoms, 1, dim_x, dim_y
            tile_shape = list(x.size())
            tile_shape[3] = self.output_dim * self.output_atoms     # batch_size, input_dim, input_atoms, output_dim*output_atoms, dim_x, dim_y
            x = x.expand(tile_shape)                                # batch_size, input_dim, input_atoms, output_dim*output_atoms, dim_x, dim_y
            x = torch.sum(x * self.weights, dim=2)                  # batch_size, input_dim, output_dim*output_atoms
        elif self.type == 2:
            act = x.norm(dim=2, keepdim=True)
            act_sorted = act.view(x_sh[0],-1).sort(-1, descending=True)
            #act_sorted = act_sorted[1].unsqueeze(-1).repeat(1,1,x_sh[2])
            #x = torch.gather(x.view(x_sh[0],x_sh[1],-1), 1, act_sorted).view(x_sh)

            x.unsqueeze_(3)                                         # batch_size, input_dim, input_atoms, 1, dim_x, dim_y
            tile_shape = list(x.size())
            tile_shape[3] = self.output_dim * self.output_atoms     # batch_size, input_dim, input_atoms, output_dim*output_atoms, dim_x, dim_y
            x = x.expand(tile_shape)                                # batch_size, input_dim, input_atoms, output_dim*output_atoms, dim_x, dim_y
            x = torch.sum(x * self.weights, dim=2)                  # batch_size, input_dim, output_dim*output_atoms
            
            act_sorted = act_sorted[1].view(x_sh[0],-1,1,1,1).expand(x.shape)
            x = torch.gather(x, 1, act_sorted)
            x = x[:,int(x_sh[1]/2):,:,:,:]
            #x = torch.index_select(x.view(-1,tile_shape[3],1,1), 0, act_sorted[1].view(-1)).view(x.shape)
        else:
            x = x.view(x_sh[0]*x_sh[1], x_sh[2], x_sh[3], x_sh[4])  # batch_size*input_dim, input_atoms, dim_x, dim_y
            x = self.conv(x)                                        # batch_size*input_dim, output_dim*output_atoms, out_dim_x, out_dim_y
            #x = self.batchnorm(x)
            
        #votes = x.view(x_sh[0], x_sh[1], self.output_dim, self.output_atoms, x.size(-2), x.size(-1))
        votes = x.view(x_sh[0], -1, self.output_dim, self.output_atoms, x.size(-2), x.size(-1))
                                                                    # batch_size, input_dim, output_dim1, output_atoms, dim_x, dim_y
        biases_replicated = self.bias.repeat([1,1,x.size(-2),x.size(-1)])
                                                                    # output_dim, output_atoms, dim_x, dim_y
        logit_shape = list(votes.size())
        logit_shape.pop(3)                                          # batch_size, input_dim, output_dim, dim_x, dim_y

        return _routing(votes=votes, biases=biases_replicated, logit_shape=logit_shape, num_routing=self.num_routing, device=self.device)


def _routing(votes, biases, logit_shape, num_routing, device):
    votes_t_shape = [3, 0, 1, 2, 4, 5]
    r_t_shape = [1, 2, 3, 0, 4, 5]
                                                                    # votes: batch_size, input_dim, output_dim, output_atoms, (dim_x, dim_y)
    
    votes_trans = votes.permute(votes_t_shape)                      # output_atoms, batch_size, input_dim, output_dim, (dim_x, dim_y)
    votes_trans_stopped = votes_trans.clone().detach()

    logits = Variable(torch.zeros(logit_shape, device=device), requires_grad=False)
                                                                    # batch_size, input_dim, output_dim, (dim_x, dim_y)
    for i in range(num_routing):
        route = F.softmax(logits, 2)                                # batch_size, input_dim, output_dim, (dim_x, dim_y)

        if i == num_routing - 1:
            preactivate_unrolled = route * votes_trans              # output_atoms, batch_size, input_dim, output_dim, (dim_x, dim_y)
            preact_trans = preactivate_unrolled.permute(r_t_shape)  # batch_size, input_dim, output_dim, output_atoms, (dim_x, dim_y)
            preactivate = torch.sum(preact_trans, dim=1) + biases   # batch_size, output_dim, output_atoms, (dim_x, dim_y)
            activation = _squash(preactivate)                       # squashing of "output_atoms" dimension
        else:
            preactivate_unrolled = route * votes_trans_stopped      # output_atoms, batch_size, input_dim, output_dim, (dim_x, dim_y)
            preact_trans = preactivate_unrolled.permute(r_t_shape)  # batch_size, input_dim, output_dim, output_atoms, (dim_x, dim_y)
            preactivate = torch.sum(preact_trans, dim=1) + biases   # batch_size, output_dim, output_atoms, (dim_x, dim_y)
            activation = _squash(preactivate)                       # squashing of "output_atoms" dimension

            act_3d = activation.unsqueeze_(1)                       # batch_size, 1, output_dim, output_atoms, (dim_x, dim_y)
            distances = torch.sum(votes * act_3d, dim=3)            # batch_size, input_dim, output_dim, (dim_x, dim_y)
            logits = logits + distances                             # batch_size, input_dim, output_dim, (dim_x, dim_y)
            
    return activation                                               # batch_size, output_dim, output_atoms, (dim_x, dim_y)

def _squash(input_tensor):
    norm = torch.norm(input_tensor, p=2, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
