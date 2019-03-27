'''
Created on Jan 14, 2019

@author: jens
'''

from capsnet import MSELossWeighted, CapsNet
import util

import torch
import torch.nn as nn
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
from se3_geodesic_loss import SE3GeodesicLoss
#from sklearn.datasets.lfw import _load_imgs

torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
random.seed(1991)
np.random.seed(1991)
torch.set_printoptions(precision=3, threshold=5000, linewidth=180)

import torch.utils.data as data
import layers

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='CapsNet')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--batch_size2', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=5000000)
    parser.add_argument('--lr',help='learning rate',type=float,nargs='?',const=0,default=None,metavar='PERIOD')    
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--normalize',help='Normalize rotation part of generated labels [0-2]',type=int,nargs='?',const=1,default=None,metavar='PERIOD')    
    parser.add_argument('--jit', action='store_true', help='Enable pytorch jit compilation')
    parser.add_argument('--disable_recon', action='store_true', help='Disable Reconstruction')
    parser.add_argument('--load_loss',help='Load prev loss',type=int,nargs='?',const=1000,default=None,metavar='PERIOD')    
    parser.add_argument('--pretrained',help='load pretrained epoch',type=int,nargs='?',const=-1,default=None,metavar='PERIOD')    
    parser.add_argument('--recon_factor', type=float, default=2e-2, metavar='N', help='use reconstruction loss or not')
    parser.add_argument('--brightness', type=float, default=0, metavar='N', help='apply random brightness to images')
    parser.add_argument('--contrast', type=float, default=0, metavar='N', help='apply random contrast to images')
    parser.add_argument('--noise',help='Add noise value',type=float,default=.3,metavar='N')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='num of workers to fetch data')
    parser.add_argument('--patience', type=int, default=20, metavar='N', help='Scheduler patience')
    parser.add_argument('--dataset', type=str, default='images', metavar='N', help='dataset options: images,three_dot_3d')
    parser.add_argument('--loss', type=str, default='MSE', metavar='N', help='loss options: MSE,GEO')
    args = parser.parse_args()
    
    time_dump = int(time.time())

    """
    Load training data
    """
    #data_rep = 0 if args.loss == 'MSE' else 1
    loss_weight = None
    if args.dataset == 'three_dot':
        train_dataset = util.myTest(width=10, sz=5000, img_type=args.dataset, transform=transforms.Compose([transforms.ToTensor(),]))
        test_dataset = util.myTest(width=10, sz=100, img_type=args.dataset, rnd=True, transform=transforms.Compose([transforms.ToTensor(),]), max_z=train_dataset.max_z, min_z=train_dataset.min_z)
    elif args.dataset == 'three_dot_3d':
        train_dataset = util.myTest(width=50, sz=5000, img_type=args.dataset, transform=transforms.Compose([transforms.ToTensor(),]))
        test_dataset = util.myTest(width=50, sz=100, img_type=args.dataset, rnd=True, transform=transforms.Compose([transforms.ToTensor(),]), max_z=train_dataset.max_z, min_z=train_dataset.min_z)
    else:
        train_dataset = util.MyImageFolder(root='../../data/{}/train/'.format(args.dataset), transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), data_rep=args.loss)
        test_dataset = util.MyImageFolder(root='../../data/{}/test/'.format(args.dataset), transform=transforms.ToTensor(), target_transform=transforms.ToTensor(), data_rep=args.loss)
        loss_weight = 1
    #else:
    #    train_dataset = util.myTest(width=20, sz=5000, img_type=args.dataset) #, transform=transforms.Compose([transforms.ToTensor(),]))
    #    test_dataset = util.myTest(width=20, sz=100, img_type=args.dataset, rnd=True) #, transform=transforms.Compose([transforms.ToTensor(),]), max_z=train_dataset.max_z, min_z=train_dataset.min_z)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=False)

    sup_iterator = train_loader.__iter__()
    test_iterator = test_loader.__iter__()
    _, imgs, labels = sup_iterator.next()
    sup_iterator = train_loader.__iter__()

    if loss_weight is not None:
        loss_weight = torch.ones(labels.size(1))
        loss_weight[6:9] *= 5
    else:
        loss_weight = torch.ones(labels.size(1))

    """
    Setup model, load it to CUDA and make JIT compilation
    """
    normalize = 0
    if args.normalize:
        normalize = args.normalize
    imgs = imgs[:2]
    labels = labels[:2]
    lambda_ = 0.9 if args.pretrained else 1e-3
    model = CapsNet(labels.shape[1], img_shape=imgs[0].shape, dataset=args.dataset, data_rep=args.loss, normalize=normalize, lambda_=lambda_)
    model(imgs, labels, disable_recon=args.disable_recon)

    if not args.disable_cuda and torch.cuda.is_available():
        model.cuda()



    """
    if args.jit:
        dummy1 = torch.rand(args.batch_size,4,100,100).float()
        dummy2 = torch.rand(args.batch_size,12).float()
        if not args.disable_cuda:
            dummy1 = dummy1.cuda()
            dummy2 = dummy2.cuda()
        model(lambda_, dummy1, dummy2)
        model = torch.jit.trace(model, (lambda_, dummy1, dummy2), check_inputs=[(lambda_, dummy1, dummy2)])
    """

    print("# parameters:", sum(param.numel() for param in model.parameters()))


    """
    Construct optimizer, scheduler, and loss functions
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr if args.lr else 1e-3, betas=(0.99, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr if args.lr else 1e-3, momentum=0.99)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr if args.lr else 1e-3, momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
    
    if args.loss == 'MSE':
        caps_loss = MSELossWeighted(torch.tensor([1.,1.,1.,1.,1.,1.,5.,5.,5.,1.]).cuda())
        #caps_loss = MSELossWeighted(torch.ones(labels.size(1)).cuda())
    else:
        #caps_loss = geodesic_loss()
        caps_loss = SE3GeodesicLoss.apply
    recon_loss = nn.MSELoss(reduction='sum')

    """
    Loading weights of previously saved states and optimizer state
    """
    if args.pretrained is not None:
        util.load_pretrained(model, optimizer, args.pretrained, args.lr, not args.disable_cuda)


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
    #setting_logger.log(str(args))



    """
    If starting from scratch, ramp learning rate
    """
    """
    ramp = 0
    if not args.pretrained:
        ramp = 10
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        new_lr = lr/100.
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        ramp_step = (lr-new_lr)/ramp
    """

    steps = len(train_dataset) // args.batch_size
    steps_test = len(test_dataset) // args.batch_size
    for epoch in range(args.num_epochs):

        print("Epoch {}".format(epoch))

        """
        Training Loop
        """
        model.train()
        #model.capsules.bn1.training = False
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

                out_labels, recon = model(imgs, labels, disable_recon=args.disable_recon)

                if not args.disable_recon:
                    recon = recon.view_as(imgs)

                #for param in model.image_decoder.parameters():
                #    param.requires_grad = False
                

                """ LOSS CALCULATION """
                loss = caps_loss(out_labels, labels)

                if not args.disable_recon:
                    add_loss = recon_loss(recon, imgs)
                    loss += args.recon_factor * add_loss
                    loss_recon += args.recon_factor*add_loss.data.cpu().item() / args.batch_size

                loss.backward()
                
                optimizer.step()

                #for param in model.image_decoder.parameters():
                #    param.requires_grad = True

                loss /= args.batch_size

                """
                Logging
                """
                meter_loss.add(loss.data.cpu().item())
                #print(loss.item()-dae_loss.item(), dae_loss.item())
                medErr, xyzErr = util.get_error(out_labels.data.cpu(), labels.data.cpu())
                medErrAvg.add(medErr)
                xyzErrAvg.add(xyzErr)
                if not args.disable_recon:
                    #unsup_loss = capsloss(out_labels_unsup.view(labels_unsup.shape[0], -1)[:,:labels_unsup.shape[1]], labels_unsup)
                    #meter_loss_unsup.add(unsup_loss.data)
                    pbar.set_postfix(loss=meter_loss.value()[0], recon_=recon.sum().data.cpu().item())
                else:
                    pbar.set_postfix(loss=meter_loss.value()[0], medErr=medErrAvg.value()[0], xyzErr=xyzErrAvg.value()[0])
                pbar.update()
                
            """
            Test Loop
            """
            
            test_loss = 0
            test_loss_recon = 0
            model.eval()
            if args.loss=='MSE':
                model.target_decoder.training = True
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
    
        
                    out_labels, recon = model(imgs, labels, disable_recon=args.disable_recon)
        
                    if not args.disable_recon:
                        recon = recon.view_as(imgs)
        
                    loss = caps_loss(out_labels, labels) / args.batch_size
                    test_loss += loss.data.cpu().item()
                    if not args.disable_recon:
                        add_loss = args.recon_factor * recon_loss(recon, imgs) / args.batch_size
                        loss += add_loss
                        test_loss_recon += add_loss.data.cpu().item()
                
            #torch.enable_grad()
            test_loss /= steps_test
            test_loss_recon /= steps_test
            print("Test loss: , Test loss recon: {}".format(test_loss_recon)) #(test_loss, test_loss_recon))
            


            """
            All train data processed: Do logging
            """
            loss = meter_loss.value()[0]
            loss_recon /= steps
            train_loss_logger.log(epoch + epoch_offset, loss-loss_recon, name='loss')
            test_loss_logger.log(epoch + epoch_offset, test_loss, name='loss')

            """
            loss_relation = loss_recon/(loss-loss_recon)
            if loss_relaparam_group['lr'] = tion > 0.25 and epoch>15:
                fac = 0.25/loss_relation
                print("Loss relation = {}. Recon-factor reduced from {} to {}".format(loss_relation, args.recon_factor, args.recon_factor*fac))
                args.recon_factor *= fac
            """

            if not args.disable_recon:
                ground_truth_logger_left.log(make_grid(imgs, nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
                reconstruction_logger_left.log(make_grid(recon.data, nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
                train_loss_logger.log(epoch + epoch_offset, loss_recon, name='recon')
                test_loss_logger.log(epoch + epoch_offset, test_loss_recon, name='recon')
            
            with open("loss.log", "a") as myfile:
                myfile.write(str(loss) + '\n')

            #print("Epoch{} Train loss:{:4} target: {:4}, {:4}, {:4}, {:4}".format(epoch, loss, out_labels[0,0], out_labels[0,1], out_labels[0,2], out_labels[0,3]))


            """
            Scheduler
            """
            scheduler.step(loss)
            
            #if epoch==3:
            #    for param_group in optimizer.param_groups:
            #        param_group['lr'] = param_group['lr'] * 10.
                    
                
            """
            if ramp > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] + ramp_step
                ramp -= 1
            """

            time_now = int(time.time())
            if (time_now-time_dump) > 60*5: # dump every 15 minutes
                time_dump = time_now
                """
                Save model and optimizer states
                """
                model.cpu()
                torch.save(model.state_dict(), "./weights/model_{}.pth".format(epoch))
                if not args.disable_cuda:
                    model.cuda()
                torch.save(optimizer.state_dict(), "./weights/optim.pth")
                
                """
                kaj = model.state_dict()
                kaj.keys()
                kaj.pop('capsules.bn1.weight')
                kaj.pop('capsules.bn1.bias')
                kaj.pop('capsules.bn1.running_mean')
                kaj.pop('capsules.bn1.running_var')
                kaj.pop('capsules.bn1.num_batches_tracked')
                torch.save(kaj, "./weights/model_kaj.pth")
                """
