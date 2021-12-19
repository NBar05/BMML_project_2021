import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_feats, out_feats, stride=1, activate_prepres=False):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.stride = stride
        self.activate = activate_prepres
        
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(in_feats),
            nn.LeakyReLU(0.1)
        )
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_feats),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1)
        )
        self.avg_pool = nn.AvgPool2d(stride)
        
        if in_feats != out_feats:
            self.conv_match = nn.Conv2d(in_feats, out_feats, kernel_size=1, stride=1)
        
    def forward(self, x):
        if self.activate:
            x = self.bn_relu(x)
            x_orig = x
        else:
            x_orig = x
            x = self.bn_relu(x)
        
        x = self.conv_block(x)
        if self.stride != 1:
            x_orig = self.avg_pool(x_orig)
        if self.in_feats != self.out_feats:
            x_orig = self.conv_match(x_orig)
            
        return x_orig + x
        

class MadryLabCIFARModel(nn.Module):
    """
        Reproduces architecture of model used by MadryLab (ResNet-W32-10) for creating secret model trained on CIFAR-10 
        Refer to: https://github.com/MadryLab/cifar10_challenge/blob/master/model.py
        
        Paramerers: 
            n_classes (int): Number of output classes
    """
    def __init__(self, n_classes=10):
        super().__init__()
        
        feats = [16, 160, 320, 640]
        
        self.layers = [nn.Conv2d(3, feats[0], kernel_size=3, stride=1, padding=1),]
        self.layers += [ResidualBlock(feats[0], feats[1], stride=1, activate_prepres=True)]
        self.layers += [ResidualBlock(feats[1], feats[1]) for i in range(5)]
        
        self.layers += [ResidualBlock(feats[1], feats[2], stride=2)]
        self.layers += [ResidualBlock(feats[2], feats[2]) for i in range(5)]
        
        self.layers += [ResidualBlock(feats[2], feats[3], stride=2)]
        self.layers += [ResidualBlock(feats[3], feats[3]) for i in range(5)]
        self.layers += [
            nn.BatchNorm2d(feats[3]), 
            nn.LeakyReLU(0.1), 
            nn.AdaptiveAvgPool2d(1), 
        ]
        self.linear = nn.Linear(feats[3], n_classes)
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.linear(self.layers(x)[...,0,0])
    
    
    
class MadryLabMNISTModel(nn.Module):
    """
        Reproduces architecture of models used by MadryLab for creating secret model trained on MNIST
        Refer to (original Tensorflow impl): https://github.com/MadryLab/mnist_challenge/blob/master/model.py
        
        Parameters:
            n_classes (int): Number of output classes
    """
    def __init__(self, n_classes=10):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(7*7*64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
        
    def forward(self, x):
        return self.layers(x)
    