# -*- coding: utf-8 -*-

'''
The Capsules layer.
@author: Yuxian Meng
'''
# TODO: use less permute() and contiguous()

import model.capsules as mcaps
import model.util as util

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

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)

def reset_meters():
    #meter_accuracy.reset()
    meter_loss.reset()
    #confusion_meter.reset()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=500)
    #parser.add_argument('--lr', type=float, default=2e-5)
    #parser.add_argument('--lr-forced', action='store_true')
    #parser.add_argument('--lr',help='learning',type=float,default=None,metavar='PERIOD')    
    parser.add_argument('--lr',help='learning rate',type=float,nargs='?',const=0,default=None,metavar='PERIOD')    
    #parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--disable-recon', action='store_true',
                        help='Disable Reconstruction')
    parser.add_argument('--bright-contrast', action='store_true',
                        help='Add random brightness and contrast')
    parser.add_argument('--disable-encoder', action='store_true',
                        help='Disable Encoding')
    parser.add_argument('--load-loss',help='Load prev loss',type=int,nargs='?',const=1000,default=None,metavar='PERIOD')    
    parser.add_argument('--print_freq', type=int, default=10)
    #parser.add_argument('--pretrained', type=str, default="",
    #                    help='pretrained epoch number')
    parser.add_argument('--pretrained',help='load pretrained epoch',type=int,nargs='?',const=-1,default=None,metavar='PERIOD')    
    parser.add_argument('--num-classes', type=int, default=1, metavar='N',
                        help='number of output classes (default: 10)')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--env-name', type=str, default='main',
                        metavar='N', help='Environment name for displaying plot')
    parser.add_argument('--loss', type=str, default='spread_loss', metavar='N',
                        help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--routing', type=str, default='EM_routing', metavar='N',
                        help='routing to use: angle_routing, EM_routing')
    parser.add_argument('--recon-factor', type=float, default=1e-6, metavar='N',
                        help='use reconstruction loss or not')
    parser.add_argument('--max-lambda',help='max lambda value',type=float,default=1.,metavar='N')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()

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
    epoch_offset = 0
    if args.load_loss:
        if os.path.isfile('loss.log'):
            with open("loss.log", "r") as lossfile:
                loss_list = []
                for loss in lossfile:
                    loss_list.append(loss)
                while len(loss_list) > args.load_loss:
                    loss_list.pop(0)
                for loss in loss_list:
                    train_loss_logger.log(epoch_offset, float(loss))
                    epoch_offset += 1
                        

    #train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'}, env=args.env_name)
    #test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env=args.env_name)
    #test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'}, env=args.env_name)
    #confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
    #                                                 'columnnames': list(range(args.num_classes)),
    #                                                 'rownames': list(range(args.num_classes))}, env=args.env_name)
    ground_truth_logger_left = VisdomLogger('image', opts={'title': 'Ground Truth, left'}, env=args.env_name)
    ground_truth_logger_right = VisdomLogger('image', opts={'title': 'Ground Truth, right'}, env=args.env_name)
    reconstruction_logger_left = VisdomLogger('image', opts={'title': 'Reconstruction, left'}, env=args.env_name)
    reconstruction_logger_right = VisdomLogger('image', opts={'title': 'Reconstruction, right'}, env=args.env_name)

    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)

    setting_logger.log(str(args))

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    if not args.lr:
        learning_rate = 1e-2
    else:
        learning_rate = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) #, eps=1e-3)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    train_dataset = util.MyImageFolder(root='../../data/dumps/', transform=transforms.ToTensor(),
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

    if args.pretrained is not None:
        
        if args.pretrained == -1: # latest
            model_name = max(glob.iglob("./weights/em_capsules/model*.pth"),key=os.path.getctime)
        else:
            model_name = "./weights/em_capsules/model_{}.pth".format(args.pretrained)
            
        optim_name = "./weights/em_capsules/optim.pth"
        
        model.load_state_dict( torch.load(model_name) )
        if os.path.isfile(optim_name):
            optimizer.load_state_dict( torch.load(optim_name) )
            if args.lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr

            # Temporary PyTorch bugfix: https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        
        m = 0.8
        lambda_ = args.max_lambda

    #old_backup = 0
    #old_counter = 0
    #old_out_labels = 0
    #old_labels = 0

    with torch.cuda.device(args.gpu):
        if args.use_cuda:
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
                    if lambda_ < args.max_lambda:
                        lambda_ += 2e-1 / steps
                    if m < 0.9:
                        m += 2e-1 / steps

                    optimizer.zero_grad()

                    imgs, labels = data  # b,1,28,28; #b
                    #gaussian_noise = torch.randn(labels.shape[0], labels.shape[1], labels.shape[2]) * 0.01
                    #labels = labels.float()# + gaussian_noise

                    #labels = torch.cat((labels.float(),torch.zeros(labels.shape[0],2)),1)
                    labels = util.matMinRep_from_qvec(labels.float())
                    
                    #imgs = imgs[:,0,:,:].unsqueeze(1) # use only red channel
                    
                    imgs_two_channel_red = util.split_in_channels(imgs)

                    if args.bright_contrast:
                        for i in range(imgs_two_channel_red.shape[0]):
                            bright = random.random()*0.3 - 0.15
                            cont = 1. / (random.random()*1.5 + 0.5)
                            imgs_four_channel = np.copy(imgs_two_channel_red)
                            for j in range(imgs_four_channel.shape[1]):
                                imgs_four_channel[i,j,:,:] = util.applyBrightnessAndContrast(imgs_four_channel[i,j,:,:], bright, cont)
                        imgs, imgs_ref = Variable(torch.from_numpy(imgs_four_channel)), Variable(torch.from_numpy(imgs_two_channel_red))
                        if args.use_cuda:
                            imgs = imgs.cuda()
                            imgs_ref = imgs_ref.cuda()
                    else:
                        imgs = imgs_ref = Variable(torch.from_numpy(imgs_two_channel_red))
                        if args.use_cuda:
                            imgs = imgs.cuda()
                            imgs_ref = imgs

                    labels = Variable(labels)
                    if args.use_cuda:
                        labels = labels.cuda()

                    #_labels[(_labels[:,6] < 0).nonzero(),3:] *= -1
                    
                    out_labels, recon = model(imgs, lambda_, labels)


                    #print("labels: ",[i.item() for i in labels[0][0:7]],",",[i.item() for i in out_labels[0]])

                    imgs_sliced = imgs_ref[:,[0,3],...]
                    if not args.disable_recon:
                        recon = recon.view_as(imgs_sliced)
                    #out_labels = out_labels[:,:9]
                    
                    # Disable 3rd axis from rotation matrices
                    #labels.view(labels.shape[0],4,4)[:,2,0:3] = out_labels.data.view(labels.shape[0],4,4)[:,2,0:3]
                    
                    #loss = capsule_loss(imgs, out_labels[:,:_labels.shape[1]], _labels, recon)
                    loss = capsule_loss(imgs_sliced, out_labels.view(labels.shape[0], -1)[:,:labels.shape[1]], labels, recon)

                    loss.backward()
                    
                    if util.isnan(model.primary_caps.capsules_pose.weight):
                        print("isnan")
                    if util.isnan(model.primary_caps.capsules_activation.weight):
                        print("isnan")
                    if util.isnan(model.conv1.weight):
                        print("isnan")
                    if util.isnan(model.conv2.weight):
                        print("isnan")
                    if util.isnan(model.convcaps1.W):
                        print("isnan")
                    if util.isnan(model.convcaps2.W):
                        print("isnan")

                    nan = False
                    if not args.disable_encoder:
                        if util.isnan(model.convcaps1.W.grad) or util.isnan(model.convcaps1.beta_v.grad) or util.isnan(model.convcaps1.beta_a.grad):
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

                        if util.isnan(model.convcaps1.W):
                            print("isnan")
                            nan = True
                        if util.isnan(model.convcaps2.W):
                            print("isnan")
                            nan = True
                        if util.isnan(model.primary_caps.capsules_pose.weight):
                            print("isnan")
                            nan = True
                        if util.isnan(model.primary_caps.capsules_activation.weight):
                            print("isnan")
                            nan = True
                        if util.isnan(model.conv1.weight):
                            print("isnan")
                            nan = True
                        if util.isnan(model.conv2.weight):
                            print("isnan")
                            nan = True

                        if nan:
                            exit()
                        #else:
                        #    old_counter += 1
                        #    if old_counter == 100:
                        #        torch.save(model.state_dict(), "./weights/em_capsules/model_backup.pth")
                        #        torch.save(optimizer.state_dict(), "./weights/em_capsules/optim.pth".format(epoch))
                                #old_backup = copy.deepcopy(model.state_dict())
                        #        old_counter = 0

                        #old_out_labels = out_labels
                        #old_labels = labels

                        #meter_accuracy.add(out_labels.data, labels.data)
                        meter_loss.add(loss.data)
                        if not args.disable_recon:
                            pbar.set_postfix(loss=meter_loss.value()[0].item(), lambda_=lambda_, recon_=recon.sum().item())
                        else:
                            pbar.set_postfix(loss=meter_loss.value()[0].item(), lambda_=lambda_)
                        pbar.update()

                    

                    #ground_truth_logger.log(imgs_two_channel_red[:,0,:,:].squeeze())
                    #reconstruction_logger.log(imgs_two_channel_red[:,1,:,:].squeeze())

                if not args.disable_recon:
                    ground_truth_logger_left.log(
                        make_grid(imgs_sliced.data[:,0,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                                  range=(0, 1)).cpu().numpy())
                    ground_truth_logger_right.log(
                        make_grid(imgs_sliced.data[:,1,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                                  range=(0, 1)).cpu().numpy())
    
                    reconstruction_logger_left.log(
                        make_grid(recon.data[:,0,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                                  range=(0, 1)).cpu().numpy())
                    reconstruction_logger_right.log(
                        make_grid(recon.data[:,1,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True,
                                  range=(0, 1)).cpu().numpy())



                loss = meter_loss.value()[0]
                #acc = meter_accuracy.value()[0]

                train_loss_logger.log(epoch + epoch_offset, loss)
                #train_error_logger.log(epoch, acc)

                
                with open("loss.log", "a") as myfile:
                    myfile.write(str(loss.data.item())+'\n')

                print("Epoch{} Train loss:{:4}".format(epoch, loss))
                scheduler.step(loss)

                model.cpu()
                torch.save(model.state_dict(), "./weights/em_capsules/model_{}.pth".format(epoch))
                if args.use_cuda:
                    model.cuda()
                torch.save(optimizer.state_dict(), "./weights/em_capsules/optim.pth".format(epoch))

                reset_meters()

                
                """
                # Test
                print('Testing...')
                correct = 0
                for i, data in enumerate(test_loader):
                    imgs, labels = data  # b,1,28,28; #b
                    imgs, labels = Variable(imgs, volatile=True), Variable(labels, volatile=True)
                    if args.use_cuda:
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
                
