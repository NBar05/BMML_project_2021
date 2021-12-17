import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader

import foolbox as fb
import matplotlib.pyplot as plt


def whitebox_attack(model, dataloader, attack_name='LinfPGD', 
                    epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0], 
                    plot=True, aug_type='ODS', device=torch.device('cpu')):
    
    model.to(device)
    
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    print('Clear accuracy on images without adversarial attack', 
          fb.utils.accuracy(fmodel, images, labels))
    
    if attack_name == 'LinfPGD':
        attack = fb.attacks.LinfPGD()
    elif attack_name == 'LinfFastGradientAttack':
        attack = fb.attacks.LinfFastGradientAttack()
    elif attack_name == 'PGD':
        attack = fb.attacks.PGD()
    elif attack_name == 'CW':
        attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=100)
    else:
        print('WRONG NAME, RESTART')
        return False
    
    _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    robust_accuracy = 1 - success.cpu().numpy().mean(axis=-1)
    
    if plot:
        plt.figure(figsize=(15, 7)) # dpi=150
        plt.plot(epsilons, robust_accuracy)
        
        plt.title(attack_name + '_' + aug_type)
        plt.xlabel('attack epsilon')
        plt.ylabel('robust accuracy')
        plt.grid()
        
        # plt.yscale('log')
        
        plt.savefig(attack_name + '_' + aug_type + '.png')
        plt.show()
    
    np.save(attack_name + '_' + aug_type + '.npy', robust_accuracy)

    model.to(torch.device('cpu'))
    
    return robust_accuracy
