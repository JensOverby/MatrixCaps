import os
import scipy
import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import datasets, transforms

def load_data(cfg):
    kwargs = {'num_workers': 3, 'pin_memory': True} if cfg.use_cuda else {}
    if cfg.dataset == 'mnist':
        train_loader = Data.DataLoader(
            datasets.MNIST(cfg.path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            batch_size=cfg.batch_size, shuffle=True, **kwargs)
        test_loader = Data.DataLoader(
            datasets.MNIST(cfg.path, train=False, transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=100, shuffle=True, **kwargs)
        features = {'height':28,
                    'depth':1,
                    'num_classes':10,
                    'num_samples':60000,
                    'num_test_samples':10000}

    else:
        raise ValueError('Dataset is not supported.')

    return train_loader, test_loader, features

def learning_rate_decay(global_step, lr, cfg):
    new_lr = max(cfg.learning_rate * cfg.decay_rate**(global_step / cfg.decay_steps), 1e-6)
    return new_lr, lr == new_lr


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs) * 255.  # inverse_transform
    imgs = mergeImgs(imgs, size).astype(np.uint8)
    imgs = Image.fromarray(imgs)
    imgs.save(path)


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs

def save_image(cfg, epoch, global_step, teX, te_image, features, idx=None):
    teX = teX.cpu().data.numpy().reshape((-1, features['height'], features['height'], features['depth']))
    te_image = te_image.cpu().data.numpy().reshape((-1, features['height'], features['height'], features['depth']))
    num_pred_samples = 100
    assert teX.shape[0] == num_pred_samples
    num_test_batch = len(teX) // cfg.batch_size

    im_size = teX.shape[-2]
    im_channel = teX.shape[-3]
    save_images(teX, [10, 10], "./results/"+str(epoch)+".jpg")
    save_images(te_image, [10, 10], "./results/gt.jpg")
    print('img saved')
