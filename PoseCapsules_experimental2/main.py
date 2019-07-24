'''
Created on Jan 14, 2019

@author: jens
'''

from capsnet import CapsNet #, MSELossWeighted
import util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import numpy as np
import random
import time
import argparse
from tqdm import tqdm
#from sklearn.datasets.lfw import _load_imgs
from torch.autograd import detect_anomaly
from datasets import smallNORB, MARAHandDataset #, transform_train

#torch.manual_seed(1991)
#torch.cuda.manual_seed(1991)
#random.seed(1991)
#np.random.seed(1991)
torch.set_printoptions(precision=3, threshold=5000, linewidth=180)

import torch.utils.data as data

"""
def hookFunc(module, gradInput, gradOutput):
    print(len(gradInput))
    for v in gradInput:
        print (v)
model.register_backward_hook(hookFunc)
"""




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
    parser.add_argument('--train_recon_only_gt', action='store_true', help='Train only the image decoder with ground truth labels as input')
    parser.add_argument('--disable_loss', action='store_true', help='Disable normal loss back prop')
    parser.add_argument('--regularize', action='store_true', help='Regularize model')
    parser.add_argument('--recon_factor', type=float, default=0., metavar='N', help='use reconstruction loss or not')
    parser.add_argument('--ramp_recon', action='store_true', help='Ramp Reconstruction')
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
    elif args.dataset == 'matmul_test' or args.dataset == 'matmul':
        train_dataset = util.myTest(width=3, sz=500, img_type=args.dataset, transform=transforms.Compose([transforms.ToTensor(),]))
        test_dataset = util.myTest(width=3, sz=10, img_type=args.dataset, transform=transforms.Compose([transforms.ToTensor(),]))
        logger = util.statNothing()
    elif args.dataset[:5] == 'MNIST':
        train_dataset = datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='../../data/', train=False, transform=transforms.ToTensor())
        logger = util.statClassification(args)
    elif args.dataset == 'smallNORB':   # transforms.Resize(48),
        train_dataset = smallNORB('../../data/smallnorb/', train=True, download=True, transform=transforms.Compose([transforms.RandomCrop(64),transforms.ColorJitter(brightness=32./255, contrast=0.5),transforms.ToTensor()]))
        test_dataset = smallNORB('../../data/smallnorb/', train=False,transform=transforms.Compose([transforms.CenterCrop(64),transforms.ToTensor()]))
    elif args.dataset == 'msra':
        train_dataset = MARAHandDataset('../../data/cvpr15_MSRAHandGestureDB', 'train', 2)
        #train_dataset.test()
        test_dataset = MARAHandDataset('../../data/cvpr15_MSRAHandGestureDB', 'test', 2)
        logger = util.statJoints(args, train_dataset.scale)
    else:
        train_dataset = util.MyImageFolder(root='../../data/{}/train/'.format(args.dataset), transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
        test_dataset = util.MyImageFolder(root='../../data/{}/test/'.format(args.dataset), transform=transforms.ToTensor(), target_transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False)
    sup_iterator = train_loader.__iter__()
    test_iterator = test_loader.__iter__()
    imgs, labels = sup_iterator.next()
    sup_iterator = train_loader.__iter__()


    """
    Setup model, load it to CUDA and make JIT compilation
    """
    #imgs = imgs[:2]
    stat = []
    model = CapsNet(args, len(train_dataset) // (2*args.batch_size) + 3, stat)

    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        imgs = imgs.cuda()
    if args.jit:
        model = torch.jit.trace(model, (imgs), check_inputs=[(imgs)])
    else:
        model(imgs)
    print("# model parameters:", sum(param.numel() for param in model.parameters()))

    """
    Construct optimizer, scheduler, and loss functions
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr if args.lr else 1e-3, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr if args.lr else 1e-3, momentum=0.9, weight_decay=0.0005)
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, threshold=1e-6, verbose=True)
    caps_loss = util.CapsuleLoss(args, 1/(len(train_dataset)/args.batch_size), model)
    recon_loss = nn.MSELoss(reduction='sum')


    """
    Loading weights of previously saved states and optimizer state
    """
    if args.pretrained is not None:
        util.load_pretrained(model, optimizer, args.pretrained, args.lr, use_cuda)


    """
    Potentially, setup ramp in of reconstruction model
    """
    ramp_recon_counter = 0
    if args.ramp_recon:
        ramp_recon_counter = 6
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4

    if model.recon_factor.item() == 0. or args.ramp_recon or args.recon_factor != 0:
        if args.recon_factor == 0.:
            args.recon_factor = 0.005
        model.recon_factor = nn.Parameter(torch.tensor(args.recon_factor, device=model.recon_factor.device), requires_grad=False)
        
    #mu_factor = 0.4
    #var_factor = 1e-5 #1/25550
    dff = 4.

    """
    Logging of loss, reconstruction and ground truth
    """
    if args.load_loss:
        logger.load_loss(args.load_loss)


    i_imgs, recon = None, None
    steps = len(train_dataset) // (args.batch_size) #*5)
    steps_test = len(test_dataset) // args.batch_size
    for epoch in range(args.num_epochs):

        print("Epoch {}".format(epoch))

        """
        Training Loop
        """
        model.train()
        sup_iterator = train_loader.__iter__()

        with tqdm(total=steps) as pbar:
            logger.reset()
            torch.cuda.empty_cache()
            for _ in range(steps):

                try:
                    data = sup_iterator.next()
                except StopIteration:
                    sup_iterator = train_loader.__iter__()
                    data = sup_iterator.next()

                imgs, labels = data

                imgs = Variable(imgs)
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()




                """
                """
                optimizer.zero_grad()
                out_labels = model.capsules(imgs)


                """ LOSS CALCULATION """
                loss = 0
                if not args.disable_loss:
                    loss = caps_loss(out_labels, labels) + loss
                    logger.lossAvg.add(loss.data.cpu().item()/args.batch_size)
                if args.regularize:
                    reguloss = 0
                    for routing in model.routing_list:
                        reguloss = routing.log_sigma.norm(p=1) + reguloss
                    reguloss = model.regularize_factor * reguloss
                    logger.regularizeLossAvg.add(reguloss.data.cpu().item()/args.batch_size)
                    loss = reguloss + loss

                #loss += model.capsules.sparse.loss * 0.01

                """
                mu_loss = 0
                var_loss = 0
                mu_numel = 0
                var_numel = 0
                for module, mu_ref, var_ref in model.route_list:
                    numel = module.a.numel()
                    mu = module.a.mean()
                    var = ((module.a - mu) ** 2).sum() / numel
                    #sparse_loss = (abs(mu - 0.5) + 2.*abs(var - 0.1)) * numel + sparse_loss

                    mu_error = mu - mu_ref
                    if mu_error > 0:
                        mu_loss = mu_error * numel + mu_loss
                        mu_numel += numel

                    var_error = var - var_ref
                    if var_error < 0:
                        var_loss = -var_error * numel + var_loss
                        var_numel += numel
                if mu_numel > 0:
                    mu_loss = mu_factor * mu_loss / mu_numel
                    logger.lossSparseMu.add(mu_loss.data.cpu().item())
                    loss = mu_loss + loss
                if var_numel > 0:
                    var_loss = var_factor * var_loss / var_numel
                    logger.lossSparseVar.add(var_loss.data.cpu().item())
                    loss = var_loss + loss
                """


                if not args.disable_recon:
                    i_imgs = imgs
                    if i_imgs.shape[-1] > 100:
                        i_imgs = F.interpolate(i_imgs, size=100)
                    if args.train_recon_only_gt:
                        ins = out_labels.data.squeeze()[:,:,-1].unsqueeze(-1)
                        tmp_labels = torch.cat([labels, ins], dim=2).view_as(out_labels)
                        recon = model.image_decoder((tmp_labels,None))
                        loss = 0
                    else:
                        recon = model.image_decoder(out_labels)

                    recon = recon.view_as(i_imgs)
                    add_loss = model.recon_factor * recon_loss(recon, i_imgs)
                    loss = add_loss + loss

                    logger.reconLossAvg.add(add_loss.data.cpu().item() / args.batch_size)
                    logger.recon_sum = recon.sum().data.cpu().item()

                #torch.autograd.set_detect_anomaly(True)
                #with detect_anomaly():
                loss.backward()
                
                optimizer.step()

                
                logger.log(pbar, out_labels, labels, stat=stat)

                pbar.update()
                


            logger.endTrainLog(epoch, i_imgs, recon)


            """
            Scheduler
            """
            scheduler.step(logger.lossAvg.value()[0])
            
            if not args.disable_recon and ramp_recon_counter==0 and not args.disable_loss:
                diff_factor = logger.lossAvg.value()[0] / (logger.reconLossAvg.value()[0]*4)
                if diff_factor > 1:
                    model.recon_factor += 0.2*model.recon_factor*(1-1/diff_factor)
                else:
                    model.recon_factor -= 0.2*model.recon_factor*(1-diff_factor)
            
            if args.regularize:
                if logger.regularizeLossAvg.value()[0] != 0:
                    diff_factor = logger.lossAvg.value()[0] / (logger.regularizeLossAvg.value()[0]*dff)
                    model.regularize_factor *= diff_factor
            
            if ramp_recon_counter > 0:
                ramp_recon_counter -= 1
                if ramp_recon_counter == 1:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.5*args.lr
                elif ramp_recon_counter == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr


            time_now = int(time.time())
            if (time_now-time_dump) > 60*5: # dump every 5 minutes
                time_dump = time_now
                """
                Save model and optimizer states
                """
                model.cpu()
                torch.save(model.state_dict(), "./weights/model_{}.pth".format(epoch))
                if use_cuda:
                    model.cuda()
                torch.save(optimizer.state_dict(), "./weights/optim.pth")

                
        """
        Test Loop
        """
        model.eval()
        print()
        print("Testing...")
        with tqdm(total=steps_test) as pbar:
            logger.reset()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for _ in range(steps_test):
                    try:
                        data = test_iterator.next()
                    except StopIteration:
                        test_iterator = test_loader.__iter__()
                        data = test_iterator.next()
        
                    imgs, labels = data
                    
                    imgs = Variable(imgs)
                    if use_cuda:
                        imgs = imgs.cuda()
                        labels = labels.cuda()
    
        
                    out_labels = model.capsules(imgs)

                    """ LOSS CALCULATION """
                    if not args.disable_loss:
                        loss = caps_loss(out_labels, labels)
                        logger.lossAvg.add(loss.data.cpu().item()/args.batch_size)
                    if args.regularize:
                        reguloss = 0
                        for routing in model.routing_list:
                            reguloss = routing.log_sigma.norm(p=1) + reguloss
                        reguloss = model.regularize_factor * reguloss
                        logger.regularizeLossAvg.add(reguloss.data.cpu().item()/args.batch_size)


                    if not args.disable_recon:
                        i_imgs = imgs
                        if i_imgs.shape[-1] > 100:
                            i_imgs = F.interpolate(i_imgs, size=100)
                        recon = model.image_decoder(out_labels)
                        recon = recon.view_as(i_imgs)
                        
                        add_loss = model.recon_factor * recon_loss(recon, i_imgs)
                        logger.reconLossAvg.add(add_loss.data.cpu().item() / args.batch_size)
                        
                    logger.log(pbar, out_labels, labels, stat=stat)
                    pbar.update()
                

            """
            All train data processed: Do logging
            """
            logger.endTestLog(epoch)


