'''
Created on Jan 14, 2019

@author: jens
'''

from capsnet import MSELossWeighted, CapsNet
import util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import random
import time
import argparse
from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
#from sklearn.datasets.lfw import _load_imgs
from torch.autograd import detect_anomaly

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)
torch.set_printoptions(precision=3, threshold=5000, linewidth=180)

import torch.utils.data as data


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='CapsNet')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=5000000)
    parser.add_argument('--lr',help='learning rate',type=float,nargs='?',const=0,default=None,metavar='PERIOD')    
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--normalize',help='Normalize rotation part of generated labels [0-2]',type=int,nargs='?',const=1,default=None,metavar='PERIOD')    
    parser.add_argument('--jit', action='store_true', help='Enable pytorch jit compilation')
    parser.add_argument('--load_loss',help='Load prev loss',type=int,nargs='?',const=1000,default=None,metavar='PERIOD')    
    parser.add_argument('--pretrained',help='load pretrained epoch',type=int,nargs='?',const=-1,default=None,metavar='PERIOD')    
    parser.add_argument('--disable_recon', action='store_true', help='Disable Reconstruction')
    parser.add_argument('--recon_factor', type=float, default=0.05, metavar='N', help='use reconstruction loss or not')
    parser.add_argument('--ramp_recon', action='store_true', help='Ramp Reconstruction')
    parser.add_argument('--brightness', type=float, default=0, metavar='N', help='apply random brightness to images')
    parser.add_argument('--contrast', type=float, default=0, metavar='N', help='apply random contrast to images')
    parser.add_argument('--noise',help='Add noise value',type=float,default=.3,metavar='N')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='num of workers to fetch data')
    parser.add_argument('--patience', type=int, default=20, metavar='N', help='Scheduler patience')
    parser.add_argument('--dataset', type=str, default='images', metavar='N', help='dataset options: images,three_dot_3d')
    args = parser.parse_args()
    time_dump = int(time.time())

    
    """
    Load training data
    """
    if args.dataset == 'three_dot':
        train_dataset = util.myTest(width=10, sz=5000, img_type=args.dataset, transform=transforms.Compose([transforms.ToTensor(),]))
        test_dataset = util.myTest(width=10, sz=100, img_type=args.dataset, rnd=True, transform=transforms.Compose([transforms.ToTensor(),]), max_z=train_dataset.max_z, min_z=train_dataset.min_z)
    elif args.dataset == 'three_dot_3d':
        train_dataset = util.myTest(width=50, sz=5000, img_type=args.dataset, transform=transforms.Compose([transforms.ToTensor(),]))
        test_dataset = util.myTest(width=50, sz=100, img_type=args.dataset, rnd=True, transform=transforms.Compose([transforms.ToTensor(),]), max_z=train_dataset.max_z, min_z=train_dataset.min_z)
    else:
        train_dataset = util.MyImageFolder(root='../../data/{}/train/'.format(args.dataset), transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
        test_dataset = util.MyImageFolder(root='../../data/{}/test/'.format(args.dataset), transform=transforms.ToTensor(), target_transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=False)
    sup_iterator = train_loader.__iter__()
    test_iterator = test_loader.__iter__()
    _, imgs, labels = sup_iterator.next()
    sup_iterator = train_loader.__iter__()


    """
    Setup model, load it to CUDA and make JIT compilation
    """
    imgs = imgs[:2]
    model = CapsNet(args.dataset)
    model(imgs)
    print("# model parameters:", sum(param.numel() for param in model.parameters()))
    if not args.disable_cuda and torch.cuda.is_available():
        model.cuda()
        imgs = imgs.cuda()
    if args.jit:
        model = torch.jit.trace(model, (imgs), check_inputs=[(imgs)])


    """
    Construct optimizer, scheduler, and loss functions
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr if args.lr else 1e-3, betas=(0.99, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
    caps_loss = nn.MSELoss(reduction='sum') #MSELossWeighted(args.batch_size, 1.0, pretrained=args.pretrained)
    recon_loss = nn.MSELoss(reduction='sum')


    """
    Loading weights of previously saved states and optimizer state
    """
    if args.pretrained is not None:
        util.load_pretrained(model, optimizer, args.pretrained, args.lr, not args.disable_cuda)


    """
    Potentially, setup ramp in of reconstruction model
    """
    ramp_recon_counter = 0
    if args.ramp_recon:
        ramp_recon_counter = 6
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4

    if model.recon_factor.item() == 0. or args.ramp_recon:
        model.recon_factor = nn.Parameter(torch.tensor(args.recon_factor, device=model.recon_factor.device), requires_grad=False)
        


    """
    Logging of loss, reconstruction and ground truth
    """
    meter_loss = tnt.meter.AverageValueMeter()
    medErrAvg = tnt.meter.AverageValueMeter()
    xyzErrAvg = tnt.meter.AverageValueMeter()    
    setting_logger = VisdomLogger('text', opts={'title': 'Settings'}, env='PoseCapsules')
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'}, env='PoseCapsules')
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'}, env='PoseCapsules')
    epoch_offset = 0
    if args.load_loss:
        epoch_offset = util.load_loss(train_loss_logger, args.load_loss)
    ground_truth_logger_left = VisdomLogger('image', opts={'title': 'Ground Truth, left'}, env='PoseCapsules')
    reconstruction_logger_left = VisdomLogger('image', opts={'title': 'Reconstruction, left'}, env='PoseCapsules')



    steps = len(train_dataset) // args.batch_size
    steps_test = len(test_dataset) // args.batch_size
    for epoch in range(args.num_epochs):

        print("Epoch {}".format(epoch))

        """
        Training Loop
        """
        model.train()

        with tqdm(total=steps) as pbar:
            meter_loss.reset()
            medErrAvg.reset()
            xyzErrAvg.reset()

            loss_recon = 0
            torch.cuda.empty_cache()
            for _ in range(steps):
                try:
                    data = sup_iterator.next()
                except StopIteration:
                    sup_iterator = train_loader.__iter__()
                    data = sup_iterator.next()

                _,imgs, labels = data

                """
                Transform images and labels
                """
                if args.brightness != 0 or args.contrast != 0:
                    brightness = torch.rand(100)*args.brightness*2 - args.brightness
                    contrast = torch.rand(100)*args.contrast*2 + 1-args.contrast
                    brightness = brightness.expand_as(imgs)
                    contrast = contrast.expand_as(imgs)
                    imgs = 0.5*(1-contrast) + brightness + contrast*imgs
                    imgs = torch.clamp(imgs, 0., 1.)


                imgs = Variable(imgs)
                #labels = util.matMinRep_from_qvec(labels)
                if not args.disable_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()



                """
                """
                optimizer.zero_grad()
                out_labels = model.capsules(imgs)


                """ LOSS CALCULATION """
                loss = caps_loss(out_labels[:,0,0,0,:-1], labels)


                if not args.disable_recon:
                    #if imgs.shape[1] > 1:
                    #    """ greyscale """
                    #    i_imgs = (0.2989*imgs.data[:,0,:,:] + 0.5870*imgs.data[:,1,:,:] + 0.1140*imgs.data[:,2,:,:]).unsqueeze(1)
                    #else:
                    i_imgs = imgs
                    if i_imgs.shape[-1] > 100:
                        i_imgs = F.interpolate(i_imgs, size=100)
                    recon = model.image_decoder(out_labels)
                    recon = recon.view_as(i_imgs)
                    add_loss = recon_loss(recon, i_imgs)
                    loss += model.recon_factor * add_loss
                    loss_recon += model.recon_factor.item()*add_loss.data.cpu().item() / args.batch_size


                torch.autograd.set_detect_anomaly(True)
                with detect_anomaly():
                    loss.backward()
                
                optimizer.step()

                
                """
                Logging
                """
                loss /= args.batch_size
                meter_loss.add(loss.data.cpu().item())
                medErr, xyzErr = util.get_error(out_labels[:,0,0,0,:-1].data.cpu(), labels.data.cpu())
                medErrAvg.add(medErr)
                xyzErrAvg.add(xyzErr)
                if not args.disable_recon:
                    pbar.set_postfix(loss=meter_loss.value()[0], AngErr=medErrAvg.value()[0], xyzErr=xyzErrAvg.value()[0], recon_=recon.sum().data.cpu().item())
                else:
                    pbar.set_postfix(loss=meter_loss.value()[0], AngErr=medErrAvg.value()[0], xyzErr=xyzErrAvg.value()[0])
                pbar.update()



            if not args.disable_recon:
                ground_truth_logger_left.log(make_grid(i_imgs, nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
                reconstruction_logger_left.log(make_grid(recon.data, nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())


                
            """
            Test Loop
            """
            
            test_loss = 0
            test_loss_recon = 0
            model.eval()
            
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for _ in range(steps_test):
                    try:
                        data = test_iterator.next()
                    except StopIteration:
                        test_iterator = test_loader.__iter__()
                        data = test_iterator.next()
        
                    _,imgs, labels = data
                    
                    imgs = Variable(imgs)
                    #labels = util.matMinRep_from_qvec(labels)
                    if not args.disable_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()
    
        
                    out_labels = model.capsules(imgs)
        
                    loss = caps_loss(out_labels[:,0,0,0,:-1], labels) / args.batch_size
                    test_loss += loss.data.cpu().item()

                    if not args.disable_recon:
                        #if imgs.shape[1] > 1:
                        #    """ greyscale """
                        #    i_imgs = (0.2989*imgs.data[:,0,:,:] + 0.5870*imgs.data[:,1,:,:] + 0.1140*imgs.data[:,2,:,:]).unsqueeze(1)
                        #else:
                        i_imgs = imgs
                        if i_imgs.shape[-1] > 100:
                            i_imgs = F.interpolate(i_imgs, size=100)
                        recon = model.image_decoder(out_labels)
                        recon = recon.view_as(i_imgs)
                        
                        add_loss = model.recon_factor.item() * recon_loss(recon, i_imgs) / args.batch_size
                        loss += add_loss
                        test_loss_recon += add_loss.data.cpu().item()
                
            test_loss /= steps_test
            test_loss_recon /= steps_test
            


            """
            All train data processed: Do logging
            """
            loss = meter_loss.value()[0]
            loss_recon /= steps
            train_loss_logger.log(epoch + epoch_offset, loss-loss_recon, name='loss')
            test_loss_logger.log(epoch + epoch_offset, test_loss, name='loss')

            if not args.disable_recon:
                train_loss_logger.log(epoch + epoch_offset, loss_recon, name='recon')
                test_loss_logger.log(epoch + epoch_offset, test_loss_recon, name='recon')
            
            with open("loss.log", "a") as myfile:
                myfile.write(str(loss) + '\n')



            """
            Scheduler
            """
            scheduler.step(loss)
            
            if not args.disable_recon and ramp_recon_counter==0:
                pose_loss = loss-loss_recon
                diff_factor = pose_loss / loss_recon
                model.recon_factor -= 0.1*model.recon_factor*(1-diff_factor)
            
            
            if ramp_recon_counter > 0:
                ramp_recon_counter -= 1
                if ramp_recon_counter == 1:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.5*args.lr
                elif ramp_recon_counter == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr


            time_now = int(time.time())
            if (time_now-time_dump) > 60*15: # dump every 15 minutes
                time_dump = time_now
                """
                Save model and optimizer states
                """
                model.cpu()
                torch.save(model.state_dict(), "./weights/model_{}.pth".format(epoch))
                if not args.disable_cuda:
                    model.cuda()
                torch.save(optimizer.state_dict(), "./weights/optim.pth")
