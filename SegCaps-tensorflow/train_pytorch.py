'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

#from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image

"""
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
"""

#from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss
from load_3D_data import load_class_weights, generate_train_batches, generate_val_batches

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight
        self.zero_tensor = torch.zeros(1).cuda()


    def weighted_cross_entropy_with_logits(self, logits, target):
        """
        Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        weighted logistic losses.
        """
        
        l = (1 + (self.pos_weight - 1) * target)
        return (1 - target) * logits + l * (torch.log(1 + torch.exp(-torch.abs(logits))) + torch.max(-logits, self.zero_tensor))
        
        #return targets * -logits.sigmoid().log() * pos_weight + 
        #           (1 - targets) * -(1 - logits.sigmoid()).log()
                   
    def forward(self, target, output):
        """Binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        
        epsilon = 1e-7
        output = torch.clamp(output, epsilon, 1-epsilon)
        output = torch.log(output / (1 - output))
        return self.weighted_cross_entropy_with_logits(output, target)
        
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        """
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))
    
        return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                       logits=output,
                                                        pos_weight=pos_weight)
        """

def get_loss(root, split, net, recon_wei, choice):
    if choice == 'w_bce':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = WeightedBinaryCrossEntropy(pos_class_weight)
    elif choice == 'bce':
        loss = 'binary_crossentropy'
    elif choice == 'dice':
        loss = dice_loss
    elif choice == 'w_mar':
        pos_class_weight = load_class_weights(root=root, split=split)
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=pos_class_weight)
    elif choice == 'mar':
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    else:
        raise Exception("Unknow loss_type")

    if net.find('caps') != -1:
        return {'out_seg': loss, 'out_recon': 'mse'}, {'out_seg': 1., 'out_recon': recon_wei}
    else:
        return loss, None
"""
def get_callbacks(arguments):
    if arguments.net.find('caps') != -1:
        monitor_name = 'val_out_seg_dice_hard'
    else:
        monitor_name = 'val_dice_hard'

    csv_logger = CSVLogger(join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, batch_size=arguments.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                       verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=5,verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(args, net_input_shape, uncomp_model):
    # Set optimizer from PIL import Imageloss and metrics
    opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)
    if args.net.find('caps') != -1:
        metrics = {'out_seg': dice_hard}
    else:
        metrics = [dice_hard]

    loss, loss_weighting = get_loss(root=args.data_root_dir, split=args.split_num, net=args.net,
                                    recon_wei=args.recon_wei, choice=args.loss)

    # If using CPU or single GPU
    if args.gpus <= 1:
        uncomp_model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return uncomp_model
    # If using multiple GPUs
    else:
        with tf.device("/cpu:0"):
            uncomp_model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
            model = multi_gpu_model(uncomp_model, gpus=args.gpus)
            model.__setattr__('callback_model', uncomp_model)
        model.compile(optimizer=opt, loss=loss, loss_weights=loss_weighting, metrics=metrics)
        return model
"""

def plot_training(training_history, arguments):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    if arguments.net.find('caps') != -1:
        ax1.plot(training_history.history['out_seg_dice_hard'])
        ax1.plot(training_history.history['val_out_seg_dice_hard'])
    else:
        ax1.plot(training_history.history['dice_hard'])
        ax1.plot(training_history.history['val_dice_hard'])
    ax1.set_title('Dice Coefficient')
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    if arguments.net.find('caps') != -1:
        ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_dice_hard'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['dice_hard'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()

def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs

def save_image(imgs, path):
    imgs = imgs.cpu().data.numpy().reshape((-1, imgs.size(2), imgs.size(3), 1))
    imgs = (imgs) * 255.
    imgs = mergeImgs(imgs, [1,1]).astype(np.uint8)
    imgs = Image.fromarray(imgs)
    imgs.save(path)

def train(args, model, train_list, net_input_shape):
    """
    # Compile the loaded model
    model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=u_model)
    # Set the callbacks
    callbacks = get_callbacks(args)

    # Training the network
    history = model.fit_generator(
        generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                               batchSize=args.batch_size, numSlices=args.slices, subSampAmt=args.subsamp,
                               stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data),
        max_queue_size=40, workers=4, use_multiprocessing=False,
        steps_per_epoch=10000,
        validation_data=generate_val_batches(args.data_root_dir, val_list, net_input_shape, net=args.net,
                                             batchSize=args.batch_size,  numSlices=args.slices, subSampAmt=0,
                                             stride=20, shuff=args.shuffle_data),
        validation_steps=500, # Set validation stride larger to see more of the data.
        epochs=200,
        callbacks=callbacks,
        verbose=1)
    """

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, betas=(0.99, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.05, patience=5, verbose=True)

    loss, loss_weighting = get_loss(root=args.data_root_dir, split=args.split_num, net=args.net,
                                    recon_wei=args.recon_wei, choice=args.loss)

    recon_loss = nn.MSELoss(reduction='sum')


    fit_generator = generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                               batchSize=args.batch_size, numSlices=args.slices, subSampAmt=args.subsamp,
                               stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data)

    factor = 1.

    for i, batch in enumerate(fit_generator):
        x = torch.from_numpy(batch[0][0]).float().permute(0,3,1,2) # -> (1,512,512,1)
        x = Variable(x).cuda()
        x1 = torch.from_numpy(batch[0][1]).float().permute(0,3,1,2) # -> (1,512,512,1)
        x1 = Variable(x1).cuda()
        y = torch.from_numpy(batch[1][0]).float().permute(0,3,1,2) # -> (1,512,512,1)
        y = y.cuda()
        label_recon = torch.from_numpy(batch[1][1]).float().permute(0,3,1,2) # -> (1,512,512,1)
        label_recon = label_recon.cuda()

        optimizer.zero_grad()

        out_seg, out_recon = model(x, x1)
        
        loss_a = loss['out_seg'](y, out_seg)

        if (i % 100) == 0:
            save_image(x, 'x.jpg')
            save_image(y, 'y.jpg')
            save_image(label_recon, 'label_recon.jpg')
            save_image(out_seg, 'out_seg.jpg')
            save_image(loss_a, 'loss_a.jpg')
            save_image(out_recon, 'out_recon.jpg')
        
        
        loss_a = loss_a.sum()
        loss_b = recon_loss(out_recon, label_recon)
        total_loss = loss_a + factor*loss_b

        total_loss.backward()
        optimizer.step()
        
        print("batch:", i, "loss a:",loss_a.data.item(), "loss b:",loss_b.data.item(), "check:", out_recon.sum().data.item())
     
     
     
        """
        torch.save(model.state_dict(), "model.pth")
        torch.save(optimizer.state_dict(), "optim.pth")
        
        model.load_state_dict(torch.load("model.pth"))
        optimizer.load_state_dict(torch.load("optim.pth"))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] =         
        """



    # Plot the training data collected
    #plot_training(history, args)
