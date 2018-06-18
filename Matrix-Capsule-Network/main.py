# -*- coding: utf-8 -*-

'''
The Capsules layer.
@author: Yuxian Meng
'''
# TODO: use less permute() and contiguous()

import model.capsules as mcaps

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import random
import os

import argparse
from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)



def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test-batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='number of output classes (default: 10)')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--env-name', type=str, default='main',
                        metavar='N', help='Environment name for displaying plot')
    parser.add_argument('--loss', type=str, default='spread_loss', metavar='N',
                        help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--routing', type=str, default='EM_routing', metavar='N',
                        help='routing to use: angle_routing, EM_routing')
    parser.add_argument('--use-recon', type=float, default=0.0005, metavar='N',
                        help='use reconstruction loss or not')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    use_cuda = args.use_cuda
    lambda_ = 1e-3  # TODO:find a good schedule to increase lambda and m
    m = 0.2

    A, B, C, D, E, r = 64, 8, 16, 16, args.num_classes, args.r  # a small CapsNet
    #A, B, C, D, E, r = 32, 32, 32, 32, args.num_classes, args.r  # a classic CapsNet

    model = mcaps.CapsNet(args, A, B, C, D, E, r, 2)
    capsule_loss = mcaps.CapsuleLoss(args)

    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)

    setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env=args.env_name)
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env=args.env_name)
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'}, env=args.env_name)
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env=args.env_name)
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'}, env=args.env_name)
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(args.num_classes)),
                                                     'rownames': list(range(args.num_classes))}, env=args.env_name)
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'}, env=args.env_name)
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'}, env=args.env_name)

    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    setting_logger.log(str(args))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True)

    steps, lambda_, m = len(train_dataset) // args.batch_size, 1e-3, 0.2

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        m = 0.8
        lambda_ = 0.9

    with torch.cuda.device(args.gpu):
        if use_cuda:
            print("activating cuda")
            model.cuda()

        for epoch in range(args.num_epochs):
            reset_meters()

            # Train
            print("Epoch {}".format(epoch))
            step = 0
            correct = 0
            loss = 0

            with tqdm(total=steps) as pbar:
                for data in train_loader:
                    step += 1
                    if lambda_ < 1:
                        lambda_ += 2e-1 / steps
                    if m < 0.9:
                        m += 2e-1 / steps

                    optimizer.zero_grad()

                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs), Variable(labels)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                    out_labels, recon = model(imgs, lambda_, labels)

                    recon = recon.view_as(imgs)
                    loss = capsule_loss(imgs, out_labels, labels, m, recon)

                    loss.backward()
                    optimizer.step()

                    meter_accuracy.add(out_labels.data, labels.data)
                    meter_loss.add(loss.data[0])
                    pbar.set_postfix(loss=meter_loss.value()[0], acc=meter_accuracy.value()[0])
                    pbar.update()

                loss = meter_loss.value()[0]
                acc = meter_accuracy.value()[0]

                train_loss_logger.log(epoch, loss)
                train_error_logger.log(epoch, acc)

                print("Epoch{} Train acc:{:4}, loss:{:4}".format(epoch, acc, loss))
                scheduler.step(acc)
                torch.save(model.state_dict(), "./weights/em_capsules/model_{}.pth".format(epoch))

                reset_meters()
                # Test
                print('Testing...')
                correct = 0
                for i, data in enumerate(test_loader):
                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs, volatile=True), Variable(labels, volatile=True)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()
                    out_labels, recon = model(imgs, lambda_)  # b,10,17

                    recon = imgs.view_as(imgs)
                    loss = capsule_loss(imgs, out_labels, labels, m, recon)

                    # visualize reconstruction for final batch
                    if i == 0:
                        ground_truth_logger.log(
                            make_grid(imgs.data, nrow=int(args.test_batch_size ** 0.5), normalize=True,
                                      range=(0, 1)).cpu().numpy())
                        reconstruction_logger.log(
                            make_grid(recon.data, nrow=int(args.test_batch_size ** 0.5), normalize=True,
                                      range=(0, 1)).cpu().numpy())

                    meter_accuracy.add(out_labels.data, labels.data)
                    confusion_meter.add(out_labels.data, labels.data)
                    meter_loss.add(loss.data[0])

                loss = meter_loss.value()[0]
                acc = meter_accuracy.value()[0]

                test_loss_logger.log(epoch, loss)
                test_accuracy_logger.log(epoch, acc)
                confusion_logger.log(confusion_meter.value())

                print("Epoch{} Test acc:{:4}, loss:{:4}".format(epoch, acc, loss))

