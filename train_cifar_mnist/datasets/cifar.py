import albumentations as albu
import torchvision.datasets as ds
import numpy as np
import torch


augs_cifar = {
    'train': albu.Compose([
        albu.PadIfNeeded(36,36),
        albu.RandomCrop(32,32),
        albu.HorizontalFlip(),
        albu.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])
    ]),
    'valid': albu.Compose([
       albu.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])
    ]),
    'test': albu.Compose([
        albu.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])
    ])
}


def get_cifar_dataset():
    cifar_train = ds.CIFAR10(root='/home/npatakin/Downloads/',train=True)
    cifar_train.transform = lambda x: torch.as_tensor(augs_cifar['train'](image=np.asarray(x))['image']).permute(2,0,1)
    cifar_val = ds.CIFAR10(root='/home/npatakin/Downloads/',train=False)
    cifar_val.transform = lambda x: torch.as_tensor(augs_cifar['test'](image=np.asarray(x))['image']).permute(2,0,1)
    
    return cifar_train, cifar_val


def get_mnist_dataset():
    trf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.MNIST(root='.', download=True, train=True, transform=trf)
    mnist_test  = torchvision.datasets.MNIST(root='.', download=True, train=False, transform=trf)
    return mnist_trian, mnist_test