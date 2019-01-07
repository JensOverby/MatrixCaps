import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch.autograd import Variable

class Model(torch.nn.Module):
    def __init__(self, features, cfg):
        super(Model, self).__init__()
        self.features = features
        self.cfg = cfg
        print(self.features)

        self.conv1 = nn.Conv2d(
            in_channels=self.features['depth'],
            out_channels=256,
            kernel_size=9,
            stride=1)
        nn.init.normal(self.conv1.weight.data, mean=0,std=5e-2)
        nn.init.constant(self.conv1.bias.data, val=0.1)

        """
            32 channels of convolutional 8D capsules (i.e. each primary capsule contains 8
            convolutional units with a 9 × 9 kernel and a stride of 2). Each primary capsule
            output sees the outputs of all 256 × 81 Conv1 units whose receptive fields overlap
            with the location of the center of the capsule. In total PrimaryCapsules has
            [32 × 6 × 6] capsule outputs (each output is an 8D vector) and each capsule in
            the [6 × 6] grid is sharing their weights with each other.
        """
        self.primary_cap = layers.ConvCapsuleLayer(
            output_dim=self.cfg.num_primary_caps, # 32
            input_atoms=256,
            output_atoms=8,
            num_routing=1,
            stride=2,
            kernel_size=5, # 9
            padding=0,
            use_cuda=self.cfg.use_cuda)

        self.conv_cap = layers.ConvCapsuleLayer(
            output_dim=32,
            input_atoms=8,
            output_atoms=8,
            num_routing=self.cfg.num_routing,
            stride=2,
            kernel_size=5,
            padding=0,
            use_cuda=self.cfg.use_cuda)

        """
            The final Layer (DigitCaps) has one 16D capsule per digit class and each of these
            capsules receives input from all the capsules in the layer below.
        """
        self.digit_cap = layers.ConvCapsuleLayer(
            output_dim=features['num_classes'], # 10
            input_atoms=8,
            output_atoms=16,
            num_routing=self.cfg.num_routing,
            stride=1,
            kernel_size=2,
            padding=0,
            use_cuda=self.cfg.use_cuda)

        num_pixels = features['height'] * features['height'] * features['depth']
        self.reconstruction = layers.Reconstruction(
            num_classes=features['num_classes'],
            num_atoms=16,
            layer_sizes=[512, 1024],
            num_pixels=num_pixels)

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = x.unsqueeze_(1)
        capsule1 = self.primary_cap(x)
        conv_cap = self.conv_cap(capsule1)
        digit_cap = self.digit_cap(conv_cap)
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
