# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-5-20 11:17:44
    @Description  : Configure
"""
import os
import torch
import random
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Deep Spiking Neural Networks')
parser.add_argument("--seed", type=int, default=2025, help='random seed')
parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers (default: 0)')

parser.add_argument("--data_set", type=str, default='*********', help='data path') ## input yourself dataset path
parser.add_argument("--data_type", type=str, default='CIFAR10', help='datasets')
parser.add_argument('--data_augment', action='store_true', default=True, help='whether to use data augmentation')
parser.add_argument("--num_epoch", type=int, default=150, help='number of epochs')
parser.add_argument("--batch_size", type=int, default=100, help='mini-batch')
parser.add_argument("--T", type=int, default=2, help='timesteps')
parser.add_argument('--use_plif', action='store_true', default=True, help='use LIF or PLIF neuron')
args = parser.parse_known_args()[0]


random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
torch.cuda.manual_seed_all(args.seed)  # all gpu, if using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True