import numpy as np
import torch
import random


def process_cli(parser):
    parser.add_argument('--dry-run', action='store_false', dest='log')
    parser.add_argument('--seed', type=int, default=324823217)
    parser.add_argument('--network', choices=['resnet20', 'cnn'], default='resnet20')
    parser.add_argument('--dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--zoo-step-size', type=float, default=5e-3)

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step'], default='cosine')
    parser.add_argument('--mask-shuffle-interval', type=int, default=5)

    parser.add_argument('--score', choices=['layer_wise_random'], default='layer_wise_random')
    parser.add_argument('--sparsity', type=float, default=0.)
    parser.add_argument('--sparsity-folder', type=str, default='Layer_Sparsity')
    parser.add_argument('--sparsity-ckpt', type=str, default=None)

    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--process-per-gpu', type=int, default=2)
    parser.add_argument('--master-addr', type=str, default='localhost')
    parser.add_argument('--master-port', type=str, default='29500')
    return parser

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def gen_folder_name(args, ignore = ['log']):
    def get_attr(inst, arg):
        value = getattr(inst, arg)
        if isinstance(value, float):
            return f"{value:.8f}"
        else:
            return value
    folder_name = ''
    for arg in vars(args):
        if arg in ignore:
            continue
        folder_name += f'{arg}-{get_attr(args, arg)}~'
    return folder_name[:-1]


def override_func(inst, func, func_name):
    bound_method = func.__get__(inst, inst.__class__)
    setattr(inst, func_name, bound_method)
