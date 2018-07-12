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
import glob

import argparse
from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
import pyrr

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)

debugcounter = 0

def reset_meters():
    #meter_accuracy.reset()
    meter_loss.reset()
    #confusion_meter.reset()

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        data = path.split('/')
        data = data[-1].split('_')
        data[-1] = data[-1].split('.p')[0]
        data = [float(i) for i in data]
        
        data = np.asarray(data)
        mat44 = pyrr.Matrix44.from_quaternion(data[3:])
        mat44[3,0] = data[0]
        mat44[3,1] = data[1]
        mat44[3,2] = data[2]
        target = np.asarray(mat44)
        
        #data.append(0.0)
        #data.append(0.0)
        #target = np.asarray(data)
        
        return sample, target

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test-batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr-forced', action='store_true')
    #parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--disable-recon', action='store_true',
                        help='Disable Reconstruction')
    parser.add_argument('--disable-encoder', action='store_true',
                        help='Disable Encoding')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--pretrained', type=str, default="",
                        help='pretrained epoch number')
    parser.add_argument('--num-classes', type=int, default=1, metavar='N',
                        help='number of output classes (default: 10)')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--env-name', type=str, default='main',
                        metavar='N', help='Environment name for displaying plot')
    parser.add_argument('--loss', type=str, default='spread_loss', metavar='N',
                        help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--routing', type=str, default='EM_routing', metavar='N',
                        help='routing to use: angle_routing, EM_routing')
    parser.add_argument('--recon-factor', type=float, default=0.0005, metavar='N',
                        help='use reconstruction loss or not')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

    use_cuda = args.use_cuda
    lambda_ = 1e-3  # TODO:find a good schedule to increase lambda and m
    m = 0.2

    #A, B, C, D, E, r = 64, 8, 16, 16, args.num_classes, args.r  # a small CapsNet
    A, AA, B, C, D, E, r = 32, 64, 16, 16, 16, args.num_classes, args.r  # a small CapsNet
    #A, B, C, D, E, r = 32, 32, 32, 32, args.num_classes, args.r  # a classic CapsNet

    model = mcaps.CapsNet(args, A, AA, B, C, D, E, r, h=4)
    capsule_loss = mcaps.CapsuleLoss(args)

    meter_loss = tnt.meter.AverageValueMeter()
    #meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    #confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)

    setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env=args.env_name)
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env=args.env_name)
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'}, env=args.env_name)
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env=args.env_name)
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'}, env=args.env_name)
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(args.num_classes)),
                                                     'rownames': list(range(args.num_classes))}, env=args.env_name)
    ground_truth_logger_left = VisdomLogger('image', opts={'title': 'Ground Truth, left'}, env=args.env_name)
    ground_truth_logger_right = VisdomLogger('image', opts={'title': 'Ground Truth, right'}, env=args.env_name)
    reconstruction_logger_left = VisdomLogger('image', opts={'title': 'Reconstruction, left'}, env=args.env_name)
    reconstruction_logger_right = VisdomLogger('image', opts={'title': 'Reconstruction, right'}, env=args.env_name)

    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    setting_logger.log(str(args))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True) #, eps=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)

    train_dataset = MyImageFolder(root='./data/dumps/', transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True)
    """
    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True)
    """

    steps, lambda_, m = len(train_dataset) // args.batch_size, 1e-3, 0.2

    if args.pretrained:
        
        if args.pretrained == 'latest':
            model_name = max(glob.iglob("./weights/em_capsules/model*.pth"),key=os.path.getctime)
            optim_name = max(glob.iglob("./weights/em_capsules/optim*.pth"),key=os.path.getctime)
        else:
            model_name = "./weights/em_capsules/model_{}.pth".format(args.pretrained)
            optim_name = "./weights/em_capsules/optim.pth"
        
        model.load_state_dict( torch.load(model_name) )
        optimizer.load_state_dict( torch.load(optim_name) )

        # Temporary PyTorch bugfix: https://github.com/pytorch/pytorch/issues/2830
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        
        if args.lr_forced:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
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
                    #imgs = imgs[:,0,:,:].unsqueeze(1) # use only red channel
                    
                    # Split in 2 stereo images,
                    # and merge to 2 channel, red only
                    left = imgs[:,:,:,:int(imgs.shape[-1]/2)]
                    right = imgs[:,:,:,int(imgs.shape[-1]/2):]
                    imgs_two_channel_red = np.stack([left[:,0,:,:],right[:,0,:,:]], axis=1)
                    
                    #ground_truth_logger.log(imgs_two_channel_red[:,0,:,:].squeeze())
                    #reconstruction_logger.log(imgs_two_channel_red[:,1,:,:].squeeze())
                    
                    #imgs = gaussian(imgs, True, 0.0, 0.1)
                    two_channel = torch.from_numpy(imgs_two_channel_red)
                    imgs, labels = Variable(two_channel), Variable(labels.reshape(labels.shape[0],-1).float())
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()

                    # DEBUG STUFF
                    print(debugcounter)
                    if debugcounter == 48:
                        debcounter = 48
                    debugcounter += 1

                    out_labels, recon = model(imgs, lambda_, labels)

                    recon = recon.view_as(imgs)
                    #out_labels = out_labels[:,:9]
                    
                    loss = capsule_loss(imgs, out_labels, labels, recon)

                    loss.backward()
                    
                    for q in range(len(model.primary_caps.capsules_activation)):
                        if mcaps.isnan(model.primary_caps.capsules_activation[q].weight):
                            print("isnan")
                        if mcaps.isnan(model.primary_caps.capsules_pose[q].weight):
                            print("isnan")
                    if mcaps.isnan(model.conv1.weight):
                        print("isnan")
                    if mcaps.isnan(model.conv2.weight):
                        print("isnan")
                    if mcaps.isnan(model.convcaps1.W):
                        print("isnan")
                    if mcaps.isnan(model.convcaps2.W):
                        print("isnan")

                    nan = False
                    if not args.disable_encoder:
                        if mcaps.isnan(model.convcaps1.W.grad) or mcaps.isnan(model.convcaps1.beta_v.grad) or mcaps.isnan(model.convcaps1.beta_a.grad):
                            nan = True
                            #scheduler._reduce_lr(epoch)
                            print("nan nan nan nan nan nan nan!!!!!!!!!!!!!!!!!!!!!!!")

                    """
                    elem_counter = 0
                    for elems in model.decoder.children():
                        if (elem_counter % 2) == 0:
                            elems.weight.grad *= 0.5
                        elem_counter += 1
                    """

                    if not nan:
                        optimizer.step()

                        if mcaps.isnan(model.convcaps1.W):
                            print("isnan")
                        if mcaps.isnan(model.convcaps2.W):
                            print("isnan")
                        for q in range(len(model.primary_caps.capsules_activation)):
                            if mcaps.isnan(model.primary_caps.capsules_activation[q].weight):
                                print("isnan")
                            if mcaps.isnan(model.primary_caps.capsules_pose[q].weight):
                                print("isnan")
                        if mcaps.isnan(model.conv1.weight):
                            print("isnan")
                        if mcaps.isnan(model.conv2.weight):
                            print("isnan")

                        #meter_accuracy.add(out_labels.data, labels.data)
                        meter_loss.add(loss.data[0])
                        pbar.set_postfix(loss=meter_loss.value()[0].item(), lambda_=lambda_, recon_=recon.sum().item())
                        pbar.update()


                    #ground_truth_logger.log(imgs_two_channel_red[:,0,:,:].squeeze())
                    #reconstruction_logger.log(imgs_two_channel_red[:,1,:,:].squeeze())


                ground_truth_logger_left.log(
                    make_grid(imgs.data[:,0,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                              range=(0, 1)).cpu().numpy())
                ground_truth_logger_right.log(
                    make_grid(imgs.data[:,1,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                              range=(0, 1)).cpu().numpy())

                reconstruction_logger_left.log(
                    make_grid(recon.data[:,0,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                              range=(0, 1)).cpu().numpy())
                reconstruction_logger_right.log(
                    make_grid(recon.data[:,1,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                              range=(0, 1)).cpu().numpy())



                loss = meter_loss.value()[0]
                #acc = meter_accuracy.value()[0]

                train_loss_logger.log(epoch, loss)
                #train_error_logger.log(epoch, acc)

                
                with open("loss.log", "a") as myfile:
                    myfile.write(str(loss.data.item())+'\n')

                print("Epoch{} Train loss:{:4}".format(epoch, loss))
                scheduler.step(loss)
                torch.save(model.state_dict(), "./weights/em_capsules/model_{}.pth".format(epoch))
                torch.save(optimizer.state_dict(), "./weights/em_capsules/optim.pth".format(epoch))

                reset_meters()

                
                """
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
                """
                
