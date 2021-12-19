import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
        

def get_dataloader(path, kind, batch_size=128):
    """
    Prepare dataloader for a "kind" split of Tiny ImageNet
    
    Params:
    - path: path to the dataset root
    - kind: train / valid / test

    return:
    - dataloader
    
    """
    
    path += kind
    
    if kind == 'train':
        dataloader = DataLoader(datasets.ImageFolder(path, transforms.ToTensor()), 
                                batch_size=batch_size, shuffle=True) # pin_memory=True
    elif kind in ['valid', 'test']:
        dataloader = DataLoader(datasets.ImageFolder(path, transforms.ToTensor()), 
                                batch_size=batch_size, shuffle=False) # pin_memory=True
    else:
        print('Error')
        return False
    
    return dataloader


# it was used to split val set on two parts: valid and test
# os.mkdir('tiny-imagenet-200/test')

# for i in range(200):
#     os.mkdir(f'tiny-imagenet-200/test/class_{i:03d}')

# for i in range(200):
#     for j in range(10):
#         os.replace(f'tiny-imagenet-200/valid/class_{i:03d}/{40+j:05d}.jpg', 
#                    f'tiny-imagenet-200/test/class_{i:03d}/{j:05d}.jpg')


# code for downloading MNIST - not used
# dataset_mnist_train = datasets.MNIST(
#     root="./",
#     train=True,
#     download=True,
#     transform=transforms.ToTensor()
# )
# dataset_mnist_valid_test = datasets.MNIST(
#     root="./",
#     train=False,
#     download=True,
#     transform=transforms.ToTensor()
# )
# print(dataset_mnist_train[1][0].size(), len(dataset_mnist_train), len(dataset_mnist_valid_test))

# dataset_mnist_valid, dataset_mnist_test = torch.utils.data.random_split(dataset_mnist_valid_test, [5000, 5000], 
#                                                                         generator=torch.Generator().manual_seed(42))

# data_loader_mnist_train = DataLoader(dataset_mnist_train, batch_size=128, shuffle=True)
# data_loader_mnist_valid = DataLoader(dataset_mnist_valid, batch_size=128, shuffle=False)
# data_loader_mnist_test = DataLoader(dataset_mnist_test, batch_size=128, shuffle=False)