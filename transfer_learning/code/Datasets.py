import os
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
import exp_configurations as Conf
from torch.utils.data import DataLoader



def get_data(dataset, batch_size):
    assert dataset in ['pubfig83',
                       'pubfig83-5',
                       'pubfig83-10',
                       'stanford_cars',
                       'stanford_cars-5',
                       'stanford_cars-10',
                       'cub',
                       'cub-5',
                       'cub-10',
                       'flower102',
                       'flower102-5',
                       'mit_indoor',
                       'mit_indoor-5',
                       'mit_indoor-10']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
            
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

    
    train_dir = os.path.join(Conf.DATA_DIR, dataset, 'train')
    val_dir = os.path.join(Conf.DATA_DIR, dataset, 'val')
    test_dir = os.path.join(Conf.DATA_DIR, dataset, 'test')


    train_data = Dataset.ImageFolder(train_dir, train_transform)
    val_data = Dataset.ImageFolder(val_dir, transform=test_transform) 
    test_data = Dataset.ImageFolder(test_dir, transform=test_transform) 


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
            shuffle=val_shuffle,
            num_workers=Conf.WORKERS, 
            pin_memory=True)
    
    return train_loader, val_loader, test_loader 


