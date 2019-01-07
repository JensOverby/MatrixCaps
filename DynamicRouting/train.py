import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch import optim
from utils import learning_rate_decay, save_image
from tqdm import tqdm

def train(model, train_loader, test_loader, features, cfg):
    global_step = 0
    lr = 0
    num_batch = int(features['num_samples'] / cfg.batch_size)

    for epoch in range(cfg.epoch):
        losses = 0
        acces = 0

        for step, (batch_xs, batch_ys) in enumerate(tqdm(train_loader, total=num_batch, ncols=50, leave=False, unit='b')):
            if len(batch_ys.shape) <= 1: #only assume shape is (bs,)
                one_hot = torch.FloatTensor(cfg.batch_size, features['num_classes']).zero_()
                batch_ys = batch_ys.unsqueeze_(1)
                one_hot.scatter_(1, batch_ys, 1.)
                batch_ys = one_hot
            batch_xs, batch_ys = Variable(batch_xs), Variable(batch_ys)

            if cfg.use_cuda:
                batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda()

            lr, lr_decay_finished = learning_rate_decay(global_step, lr, cfg)
            if not lr_decay_finished:
                optimizer = optim.Adam(model.parameters(), lr=lr)


            out, reconstruction_2d = model(batch_xs, batch_ys)

            classification_loss, reconstruction_loss = model.loss(batch_xs, out, reconstruction_2d, batch_ys)
            loss = 0.5*(classification_loss + reconstruction_loss)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses = losses + loss.cpu().data.item()

            global_step += 1

        if epoch % 5 == 0:
            for i, (batch_xs, batch_ys) in enumerate(test_loader):
                if len(batch_ys.shape) <= 1: #only assume shape is (bs,)
                    one_hot = torch.FloatTensor(cfg.batch_size, features['num_classes']).zero_()
                    batch_ys = batch_ys.unsqueeze_(1)
                    one_hot.scatter_(1, batch_ys, 1.)
                    batch_ys = one_hot
                batch_xs, batch_ys = Variable(batch_xs), Variable(batch_ys)
                if cfg.use_cuda:
                    batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda()
                out, _ = model(batch_xs, batch_ys)
                acc = model.classification_loss(out, batch_ys, 1)
                acces = acces + acc.cpu().data.item()

            _, reconstruction_2d = model(batch_xs, batch_ys)
            
            save_image(cfg, epoch, global_step, reconstruction_2d, batch_xs, features, idx=40)

            print('epoch is %d, training loss is %.4f, test acc is %.4f' % (epoch, losses, acces / features['num_test_samples']))
