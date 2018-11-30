# TODO: use less permute() and contiguous()

import model.capsules as mcaps
import model.util as util

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
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


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='CapsNet')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-size2', type=int, default=8)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=5000)
    parser.add_argument('--lr',help='learning rate',type=float,nargs='?',const=0,default=None,metavar='PERIOD')    
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--jit', action='store_true', help='Enable pytorch jit compilation')
    parser.add_argument('--disable-recon', action='store_true', help='Disable Reconstruction')
    parser.add_argument('--disable-dae', action='store_true', help='Disable Denoising Auto Encoder')
    parser.add_argument('--bright-contrast', action='store_true', help='Add random brightness and contrast')
    parser.add_argument('--disable-encoder', action='store_true', help='Disable Encoding')
    parser.add_argument('--load-loss',help='Load prev loss',type=int,nargs='?',const=1000,default=None,metavar='PERIOD')    
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--pretrained',help='load pretrained epoch',type=int,nargs='?',const=-1,default=None,metavar='PERIOD')    
    parser.add_argument('--num-classes', type=int, default=1, metavar='N', help='number of output classes (default: 10)')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('--env-name', type=str, default='main', metavar='N', help='Environment name for displaying plot')
    parser.add_argument('--loss', type=str, default='spread_loss', metavar='N', help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--recon-factor', type=float, default=1e-6, metavar='N', help='use reconstruction loss or not')
    parser.add_argument('--max-lambda',help='max lambda value',type=float,default=1.,metavar='N')
    parser.add_argument('--noise',help='Add noise value',type=float,default=.3,metavar='N')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='num of workers to fetch data')
    parser.add_argument('--patience', type=int, default=5, metavar='N', help='Scheduler patience')
    parser.add_argument('--lamb',help='Load prev loss',type=float,nargs='?',const=1e-3,default=None,metavar='PERIOD')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda and torch.cuda.is_available()


    """
    Setup model, load it to CUDA and make JIT compilation
    """
    A, AA, B, C, D, E, r = 128, 128, 40, 40, 40, args.num_classes, args.r  # a small CapsNet
    model = mcaps.CapsNet(args, A, AA, B, C, D, E, r, h=4)

    if args.lamb is None:
        lambda_ = torch.tensor([1e-3])
    else:
        lambda_ = torch.tensor([args.lamb])

    if args.use_cuda:
        model.cuda()
        lambda_ = lambda_.cuda()
    
    if args.jit:
        dummy1 = torch.rand(args.batch_size,4,100,100).float()
        dummy2 = torch.rand(args.batch_size,12).float()
        if args.use_cuda:
            dummy1 = dummy1.cuda()
            dummy2 = dummy2.cuda()
        model(lambda_, dummy1, dummy2)
        model = torch.jit.trace(model, (lambda_, dummy1, dummy2), check_inputs=[(lambda_, dummy1, dummy2)])
    
    #capsule_loss = mcaps.CapsuleLoss(args)
    print("# parameters:", sum(param.numel() for param in model.parameters()))


    """
    Construct optimizer and scheduler
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr if args.lr else 1e-2, amsgrad=True)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, verbose=True)


    """
    Loading weights of previously saved states and optimizer state
    """
    weight_folder = 'weights/{}'.format(args.env_name.replace(' ', '_'))
    if not os.path.isdir(weight_folder):
        os.mkdir(weight_folder)
    if args.pretrained is not None:
        
        if args.pretrained == -1: # latest
            model_name = max(glob.iglob("./weights/model*.pth"),key=os.path.getctime)
        else:
            model_name = "./weights/model_{}.pth".format(args.pretrained)
            
        optim_name = "./weights/optim.pth"
        
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

        if args.lamb is None:
            lambda_ = torch.tensor([args.max_lambda])
        if args.use_cuda:
            lambda_ = lambda_.cuda()

    if args.disable_encoder:
        model.dae_factor.data = torch.FloatTensor([1e-06])

    capsloss = nn.MSELoss(reduction='sum')

    """
    Logging of loss, reconstruction and ground truth
    """
    meter_loss = tnt.meter.AverageValueMeter()
    meter_loss_dae = tnt.meter.AverageValueMeter()
    meter_loss_unsup = tnt.meter.AverageValueMeter()
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

    ground_truth_logger_left = VisdomLogger('image', opts={'title': 'Ground Truth, left'}, env=args.env_name)
    ground_truth_logger_right = VisdomLogger('image', opts={'title': 'Ground Truth, right'}, env=args.env_name)
    reconstruction_logger_left = VisdomLogger('image', opts={'title': 'Reconstruction, left'}, env=args.env_name)
    reconstruction_logger_right = VisdomLogger('image', opts={'title': 'Reconstruction, right'}, env=args.env_name)
    setting_logger.log(str(args))


    """
    Load training data
    """
    train_dataset = util.MyImageFolder(root='../../data/dumps/', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False)

    sup_iterator = train_loader.__iter__()
    if not args.disable_dae or not args.disable_recon:
        unsup_loss_logger = VisdomPlotLogger('line', opts={'title': 'Unsup Loss'}, env=args.env_name)
        unsup_dataset = util.MyImageFolder(root='../../data/unsup/', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
        unsup_loader = torch.utils.data.DataLoader(dataset=unsup_dataset, batch_size=args.batch_size2, num_workers=args.num_workers, shuffle=True, drop_last=False)
        unsup_iterator = unsup_loader.__iter__()
    if not args.disable_dae:
        dae_loss_logger = VisdomPlotLogger('line', opts={'title': 'DAE Loss'}, env=args.env_name)



    steps = len(train_dataset) // args.batch_size
    if args.lamb is None:
        step = 2e-1 / steps
    else:
        step = (args.max_lambda - args.lamb) / steps

    """
    Training Loop
    """
    dae_loss = 0
    first_loop_unsup = True
    for epoch in range(args.num_epochs):

        print("dae_factor = {}".format(model.dae_factor.item()))

        # Train
        print("Epoch {}".format(epoch))

        with tqdm(total=steps) as pbar:
            meter_loss.reset()
            meter_loss_dae.reset()
            meter_loss_unsup.reset()

            #for data in train_loader:
            for _ in range(steps):
                try:
                    data = sup_iterator.next()
                except StopIteration:
                    sup_iterator = train_loader.__iter__()
                    data = sup_iterator.next()

                _,imgs, labels = data

                if not args.disable_dae or not args.disable_recon:
                    try:
                        index_unsup,imgs_unsup, labels_unsup = unsup_iterator.next()
                    except StopIteration:
                        first_loop_unsup = False
                        unsup_iterator = unsup_loader.__iter__()
                        index_unsup,imgs_unsup, labels_unsup = unsup_iterator.next()
                        
                    imgs = torch.cat([imgs, imgs_unsup], dim=0)
                    #imgs = imgs.view((imgs.shape[0]*imgs.shape[1],) + imgs.shape[2:])
                    labels_unsup = util.matMinRep_from_qvec(labels_unsup.float())
                    labels_unsup = Variable(labels_unsup)
                    if args.use_cuda:
                        labels_unsup = labels_unsup.cuda()



                if lambda_ < args.max_lambda:
                    lambda_ += step

                """
                Labels
                """
                labels = util.matMinRep_from_qvec(labels.float())
                labels = Variable(labels)
                if args.use_cuda:
                    labels = labels.cuda()


                """
                Prepare input
                """
                left = imgs[:,[0,2],:,:int(imgs.shape[-1]/2)]
                right = imgs[:,[0,2],:,int(imgs.shape[-1]/2):]
                imgs = torch.stack([left[:,:,:,2:97],right[:,:,:,2:97]], dim=3).view(imgs.shape[0],2,100, 190)
                imgs_ref = torch.stack([left[:,0,:,2:97],right[:,1,:,2:97]], dim=1)
                del left
                del right
                #imgs_stereo = np.stack([left[:,:,:,2:97],right[:,:,:,2:97]], axis=1)
                
                
                
                #imgs_two_channel_red = util.split_in_channels(imgs)
                #imgs_two_color = imgs[:,[0,2],:,5:195]

                if args.bright_contrast:
                    for i in range(imgs.shape[0]):
                        bright = random.random()*0.3 - 0.15
                        cont = 1. / (random.random()*1.5 + 0.5)
                        #imgs_four_channel = np.copy(imgs)
                        for j in range(imgs.shape[1]):
                            imgs[i,j,:,:] = util.applyBrightnessAndContrast(imgs[i,j,:,:], bright, cont)
                    #imgs, imgs_ref = Variable(torch.from_numpy(imgs_four_channel)), Variable(torch.from_numpy(imgs_two_channel_red))
                    #if args.use_cuda:
                    #    imgs = imgs.cuda()
                    #    imgs_ref = imgs_ref.cuda()

                imgs = Variable(imgs)
                imgs_ref = Variable(imgs_ref)
                if args.use_cuda:
                    imgs = imgs.cuda()
                    imgs_ref = imgs_ref.cuda()



                """
                """
                optimizer.zero_grad()

                #if not args.disable_dae:
                #    out_labels, recon, dae_loss = model(lambda_, imgs, True)
                #    dae_loss *= 1e-6

                out_labels, recon, dae_loss = model(lambda_, imgs, not args.disable_dae)

                if not args.disable_dae or not args.disable_recon:
                    out_labels_unsup = out_labels[args.batch_size:]
                    out_labels = out_labels[:args.batch_size]
                    

                #imgs_sliced = imgs_ref[:,0,:,:]
                if not args.disable_recon:
                    recon = recon.view_as(imgs_ref)
                
                #val = out_labels[:,:labels.shape[1]].view(labels.shape[0], 4, 3)[:,:3,2].data
                #ref = labels.view(labels.shape[0], 4, 3)[:,:3,2]
                #pos_error = val - ref
                #labels.view(labels.shape[0], 4, 3)[:,:3,2] = ref - 3*pos_error
                
                """ LOSS CALCULATION """
                if args.disable_encoder or labels is None:
                    caps_loss = torch.zeros(1).cuda()
                else:
                    caps_loss = capsloss(out_labels.view(labels.shape[0], -1)[:,:labels.shape[1]], labels)
                if not args.disable_recon:
                    recon_loss = capsloss(recon, imgs_ref)
                    caps_loss += args.recon_factor * recon_loss
                #caps_loss = capsule_loss(imgs_ref, out_labels.view(labels.shape[0], -1)[:,:labels.shape[1]], labels, recon)
                dae_loss *= model.dae_factor.data.item()
                loss = caps_loss + dae_loss



                loss.backward()
                
                optimizer.step()
                """
                """


                """
                Logging
                """
                #print(loss.item()-dae_loss.item(), dae_loss.item())
                meter_loss.add(caps_loss.data)
                if not args.disable_dae or not args.disable_recon:
                    unsup_loss = capsloss(out_labels_unsup.view(labels_unsup.shape[0], -1)[:,:labels_unsup.shape[1]], labels_unsup)
                    meter_loss_unsup.add(unsup_loss.data)
                    if not args.disable_recon:
                        pbar.set_postfix(capsloss=meter_loss.value()[0].item(), unsuploss=meter_loss_unsup.value()[0].item(), lambda_=lambda_.item(), recon_=recon.sum().item())
                    else:
                        meter_loss_dae.add(dae_loss.data)
                        pbar.set_postfix(capsloss=meter_loss.value()[0].item(), daeloss=meter_loss_dae.value()[0].item(), unsuploss=meter_loss_unsup.value()[0].item(), lambda_=lambda_.item())
                else:
                    if not args.disable_recon:
                        pbar.set_postfix(capsloss=meter_loss.value()[0].item(), lambda_=lambda_.item(), recon_=recon.sum().item())
                    else:
                        pbar.set_postfix(capsloss=meter_loss.value()[0].item(), lambda_=lambda_.item())
                pbar.update()

            
            """
            All train data processed: Do logging
            """
            if not args.disable_recon:
                ground_truth_logger_left.log(make_grid(imgs_ref.data[:,0,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
                ground_truth_logger_right.log(make_grid(imgs_ref.data[:,1,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())

                reconstruction_logger_left.log(make_grid(recon.data[:,0,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())
                reconstruction_logger_right.log(make_grid(recon.data[:,1,:,:].unsqueeze(1), nrow=int(args.batch_size ** 0.5), normalize=True, range=(0, 1)).cpu().numpy())

            loss = meter_loss.value()[0] # + meter_loss_dae.value()[0]
            train_loss_logger.log(epoch + epoch_offset, loss.item())
            
            with open("loss.log", "a") as myfile:
                myfile.write(str(loss.item())+'\n')

            print("Epoch{} Train loss:{:4}".format(epoch, loss.item()))


            """
            Scheduler
            """
            #scheduler.step(loss)

            if not args.disable_dae or not args.disable_recon:
                unsup_loss_logger.log(epoch + epoch_offset, meter_loss_unsup.value()[0].item())
                if not args.disable_dae:
                    dae_loss_logger.log(epoch + epoch_offset, meter_loss_dae.value()[0].item())
                    factor = 0.02 * args.batch_size/5.
                    if not args.disable_encoder and loss < factor:
                        model.dae_factor.data *= 1.1
                    """
                    else:
                        rel = loss.item() / meter_loss_dae.value()[0].item()
                        error = rel - 5.
                        if error > 5:
                            error = 5
                        if error < -5:
                            error = -5
                        model.dae_factor.data += 0.1 * error * model.dae_factor.data
                        #model.dae_factor *= 0.99
                    """

            """
            Save model and optimizer states
            """
            model.cpu()
            torch.save(model.state_dict(), "./weights/model_{}.pth".format(epoch))
            if args.use_cuda:
                model.cuda()
            torch.save(optimizer.state_dict(), "./weights/optim.pth".format(epoch))
