import argparse

def config_setting():
    ############################
    #    hyper parameters      #
    ############################
    cfg = argparse.ArgumentParser(description='PyTorch MNIST Capsnet Example')

    cfg.add_argument('--m_plus', type=float, default=0.9, help='the parameter of m plus')
    cfg.add_argument('--m_minus', type=float, default=0.1, help='the parameter of m minus')
    cfg.add_argument('--lambda_val', type=float, default=0.5, help='down weight of the loss for absent digit classes')

    # for training
    cfg.add_argument('--epoch', type=int, default=50000, help='epoch')
    cfg.add_argument('--num_routing', type=int, default=3, help='number of iterations in routing algorithm')
    cfg.add_argument('--num_primary_caps', type=int, default=32, help='number of primary caps')
    boolean_flag(cfg, 'leaky', default=True, help='if use leaky routing')
    cfg.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    cfg.add_argument('--decay_rate', type=float, default=0.96, help='learning rate decay')
    cfg.add_argument('--decay_steps', type=int, default=2000, help='decay steps')
    boolean_flag(cfg, 'use_cuda', default=True, help='use cuda')
    
    cfg.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    cfg.add_argument('--path', type=str, default='../../data/mnist/', help='the path for dataset')
    cfg.add_argument('--batch_size', type=int, default=100, help='batch size')
    cfg.add_argument('--balance_factor', type=float, default=0.0005, help='classification loss + balance_factor*reconstruciton loss')
    boolean_flag(cfg, 'constrained', default=False, help='Use constrained capsules')
    args = cfg.parse_args()
    return args

def boolean_flag(cfg, name, default=False, help=None):
    dest = name.replace('-', '_')
    cfg.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    cfg.add_argument("--no-" + name, action="store_false", dest=dest)
