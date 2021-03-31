from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

def get_stat(dataset):
    if dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return mean, std


def get_transforms(dataset):
    mean, std = get_stat(dataset)
    if dataset == 'cifar100':
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
    else:
        train_transform = transforms.Compose(
                    [transforms.RandomResizedCrop(224),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                    transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

    return train_transform, test_transform


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    elif 'bull' in hostname:
        data_folder = os.environ['LOCAL_SCRATCH']
    else:
        data_folder = './../data'


    return data_folder


class DatasetInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(DatasetInstance, self).__getitem__(index)

        return img, target, index



def get_dataloaders(dataset, batch_size=128, num_workers=8, is_instance=False):
    data_folder = get_data_folder()
    train_dir = '{}/{}/train'.format(data_folder, dataset)
    val_dir = '{}/{}/val'.format(data_folder, dataset)
    test_dir = '{}/{}/test'.format(data_folder, dataset)

    train_transform, test_transform = get_transforms(dataset) 

    if is_instance:
        train_set = DatasetInstance(train_dir,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(train_dir,
                                      transform=train_transform)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    val_set = datasets.ImageFolder(val_dir,
                                 transform=test_transform)
    
    val_loader = DataLoader(val_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)
    
    test_set = datasets.ImageFolder(test_dir,
                                 transform=test_transform)
    
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    nb_class = len(os.listdir(train_dir))
    if is_instance:
        return train_loader, val_loader, test_loader, n_data, nb_class
    else:
        return train_loader, val_loader, test_loader, nb_class



class DatasetInstanceSample(datasets.ImageFolder):
    def __init__(self, root, 
                 transform=None, target_transform=None,
                 k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, 
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = len(self.classes) 
        num_samples = len(self.samples)
        label = [item[1] for item in self.imgs] 

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
       
        img, target = super().__getitem__(index)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_dataloaders_sample(dataset, batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    data_folder = get_data_folder()
    train_dir = '{}/{}/train'.format(data_folder, dataset)
    val_dir = '{}/{}/val'.format(data_folder, dataset)
    test_dir = '{}/{}/test'.format(data_folder, dataset)

    train_transform, test_transform = get_transforms(dataset)

    train_set = DatasetInstanceSample(root=train_dir,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    val_set = datasets.ImageFolder(val_dir,
                                 transform=test_transform)
    
    val_loader = DataLoader(val_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)
    
    test_set = datasets.ImageFolder(test_dir,
                                 transform=test_transform)
    
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2),
                             pin_memory=True)

    nb_class = len(os.listdir(train_dir))

    return train_loader, val_loader, test_loader, n_data, nb_class
