import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch.autograd import Variable
from collections import OrderedDict

class Reconstruction(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(Reconstruction, self).__init__()
        self.decoder = layers.make_decoder(layer_sizes)

    def forward(self, capsule_embedding, capsule_mask):
        atom_mask = capsule_mask.clone().unsqueeze(-1).expand(capsule_embedding.size())
        filtered_embedding = capsule_embedding * atom_mask
        filtered_embedding = filtered_embedding.view(filtered_embedding.size(0), -1)
        return self.decoder(filtered_embedding)

class Model(torch.nn.Module):
    def __init__(self, features, cfg):
        super(Model, self).__init__()
        self.features = features
        self.cfg = cfg
        print(self.features)

        if self.cfg.use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')        

        self.conv1 = nn.Conv2d(in_channels=self.features['depth'], out_channels=256, kernel_size=9, stride=1)
        nn.init.normal_(self.conv1.weight.data, mean=0,std=5e-2)
        nn.init.constant_(self.conv1.bias.data, val=0.1)

        layer_list = OrderedDict()


        if cfg.constrained:
            layer_list['primary'] = layers.CapsuleLayer(output_dim=self.cfg.num_primary_caps, output_atoms=8, num_routing=1,
                                                            voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': 0}, device=device)
            layer_list['conv'] = layers.CapsuleLayer(output_dim=32, output_atoms=8, num_routing=self.cfg.num_routing,
                                                            voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 5, 'padding': 0}, device=device)
            layer_list['digit'] = layers.CapsuleLayer(output_dim=features['num_classes'], output_atoms=16, num_routing=self.cfg.num_routing,
                                                            voting={'type': 'Conv2d', 'stride': 1, 'kernel_size': 2, 'padding': 0}, device=device)
        else:
            """
                32 channels of convolutional 8D capsules (i.e. each primary capsule contains 8
                convolutional units with a 9 × 9 kernel and a stride of 2). Each primary capsule
                output sees the outputs of all 256 × 81 Conv1 units whose receptive fields overlap
                with the location of the center of the capsule. In total PrimaryCapsules has
                [32 × 6 × 6] capsule outputs (each output is an 8D vector) and each capsule in
                the [6 × 6] grid is sharing their weights with each other.
            """

            layer_list['primary'] = layers.CapsuleLayer(output_dim=self.cfg.num_primary_caps, output_atoms=8, num_routing=1,
                                                            voting={'type': 'Conv2d', 'stride': 2, 'kernel_size': 9, 'padding': 0}, device=device)

            layer_list['reshape'] = layers.Reshape()


            """
                The final Layer (DigitCaps) has one 16D capsule per digit class and each of these
                capsules receives input from all the capsules in the layer below.
            """
            layer_list['digit'] = layers.CapsuleLayer(output_dim=features['num_classes'], output_atoms=16, num_routing=self.cfg.num_routing,
                                                            voting={'type': 'standard'}, device=device)

        self.capsules = nn.Sequential(layer_list)

        num_pixels = features['height'] * features['height'] * features['depth']
        self.reconstruction = Reconstruction( [features['num_classes']*16, 512, 1024, num_pixels] )


    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = x.unsqueeze_(1)
        digit_cap = self.capsules(x)
        digit_cap.squeeze_()
        return digit_cap, self.reconstruction(digit_cap, y)

    def loss(self, images, capusle_embedding, reconstruction_2d, labels, size_average=True):
        return self.margin_loss(capusle_embedding, labels, size_average), self.reconstruction_loss(images, reconstruction_2d, size_average)

    def classification_loss(self, capusle_embedding, labels, num_targets):
        logits = torch.norm(capusle_embedding, dim=-1)
        _, targets = torch.topk(labels, k=num_targets)
        _, predictions = torch.topk(logits, k=num_targets)
        return torch.sum(targets == predictions)

    def margin_loss(self, capsule_embedding, target, size_average=True):
        batch_size = capsule_embedding.size(0)

        # ||vc|| from the paper.
        v_mag = torch.norm(capsule_embedding, dim=-1)

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).cuda()
        self.cfg.m_plus = 0.9
        self.cfg.m_minus = 0.1
        max_l = torch.max(self.cfg.m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - self.cfg.m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r

        # Here the official code does not use the sum.
        #L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, reconstruction_2d, size_average):
        image_2d = images.view(images.size(0), -1)
        distance = torch.pow(reconstruction_2d - image_2d, 2)
        loss = torch.sum(distance, dim=-1)
        batch_loss = torch.mean(loss)
        balanced_loss = self.cfg.balance_factor * batch_loss
        return balanced_loss
