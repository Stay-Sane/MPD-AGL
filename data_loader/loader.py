# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-5-20 11:17:44
    @Description  : Data Loader
"""
import os
import torch
import random
import torchvision
import numpy as np
import torch.utils.data
from torchvision import transforms
from data_loader.data_augment import Cutout, CIFAR10Policy, ImageNetPolicy


def getDataLoader(data_set, data_type, batch_size, data_augment, num_workers):
    data_path = data_set + '/' +data_type
    if data_type == 'CIFAR10':
        dataset = load_cifar10(data_path, batch_size, data_augment, num_workers)
    elif data_type == 'CIFAR10-DVS':
        dataset = load_cifar10DVS(data_path, batch_size, data_augment, num_workers)
    else:
        raise (ValueError('Unsupported dataset'))
    return dataset


def load_cifar10(data_path: str, batch_size: int, data_augment: bool, num_workers: int):
    if data_augment:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), CIFAR10Policy(),
             transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader



def load_cifar10DVS(data_path: str, batch_size: int, data_augment: bool, num_workers: int):
    train_dataset = Cifar10DVS(root=data_path + '/train', transform=data_augment)
    test_dataset = Cifar10DVS(root=data_path + '/test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


"From 'Temporal efficient training of spiking neural network via gradient re-weighting'"
class Cifar10DVS(torch.utils.data.Dataset):
    def __init__(self, root, transform=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.toPILImg = transforms.ToPILImage()
        self.resize = transforms.Resize(size=(48, 48))
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.toTensor(self.resize(self.toPILImg(data[t, ...]))))
        data = torch.stack(new_data, dim=0)
        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))