import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

def prepare_dataset(dataset, batch_size=128, pin_memory=False):
    from cfg import data_path
    data_path = os.path.join(data_path, dataset)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    if dataset == "mnist":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = train_transform
        train_data = datasets.MNIST(root = data_path, train = True, download = True, transform = train_transform)
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=0)
        test_data = datasets.MNIST(root = data_path, train = False, download = True, transform = test_transform)
        test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=0)
        cls_num = 10
    elif dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = train_transform)
        train_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=0, pin_memory=pin_memory)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = test_transform)
        test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers=0, pin_memory=pin_memory)
        cls_num = 10
    else:
        raise NotImplementedError
    return {
        'train': train_loader,
        'test': test_loader,
    }, cls_num
