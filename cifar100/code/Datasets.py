import os
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
import exp_configurations as Conf
from torch.utils.data import DataLoader


def get_cifar100(batch_size, fix_order=False):
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if fix_order:
        train_transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), 
             transforms.RandomCrop(32, padding=4), 
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize(mean, std)])

    
    train_dir = os.path.join(Conf.DATA_DIR, 'cifar100/train')
    val_dir = os.path.join(Conf.DATA_DIR, 'cifar100/val')
    test_dir = os.path.join(Conf.DATA_DIR, 'cifar100/test')

    train_data = Dataset.ImageFolder(train_dir, train_transform)
    val_data = Dataset.ImageFolder(val_dir, transform=test_transform) 
    test_data = Dataset.ImageFolder(test_dir, transform=test_transform)

    if fix_order:
        train_shuffle = False
        val_shuffle = False
        test_shuffle = False
    else:
        train_shuffle = True
        val_shuffle = False
        test_shuffle = False
    
    train_loader = DataLoader(train_data, 
            batch_size=batch_size, 
            shuffle=train_shuffle,
            num_workers=Conf.WORKERS, 
            pin_memory=True)
    
    val_loader = DataLoader(val_data, 
            batch_size=batch_size, 
            shuffle=val_shuffle,
            num_workers=Conf.WORKERS, 
            pin_memory=True)
    
    test_loader = DataLoader(test_data, 
            batch_size=batch_size, 
            shuffle=test_shuffle,
            num_workers=Conf.WORKERS, 
            pin_memory=True)

    return train_loader, val_loader, test_loader


