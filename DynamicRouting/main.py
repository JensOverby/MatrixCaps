import os
import numpy as np
import random
from config import config_setting
from model import Model
from utils import load_data
from train import train
from torch import nn

if __name__ == '__main__':
    cfg = config_setting()
    train_loader, test_loader, features = load_data(cfg)
    model = Model(features, cfg)
    if cfg.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model = model.cuda()
    print(model)
    train(model, train_loader, test_loader, features, cfg)
