import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader

import foolbox as fb
import matplotlib.pyplot as plt


def whitebox_attack(model, dataloader, attack_name='LinfPGD', 
                    epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0], 
                    plot=True, save=True, aug_type='ODS', device=torch.device('cpu')):
    """
    Function for launching whitebox attacks (utilizing model)
    
    Params:
    - model: the trained model we want to attack
    - dataloader: data set for attack
    - attack_name: name of implemented attack
    - epsilons: how much we can change the initial picture
    - plot: plot picture or not
    - save: save accuracy array or not
    - aug_type: how to name plot and/or array
    - device: cpu or cuda
    
    Return:
    - accuracy under attack
    
    """
    
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
    
    if save:
        np.save(attack_name + '_' + aug_type + '.npy', robust_accuracy)

    model.to(torch.device('cpu'))
    
    return robust_accuracy


def blackbox_attack_steps(model, dataloader, attack_name='Boundary', 
                          epsilons=[1], steps=[1000, 3000, 5000, 10000], 
                          plot=True, save=True, aug_type='ODS', device=torch.device('cpu')):
    """
    Function for launching blackbox attacks (utilizing model) for several varians of steps
    
    Params:
    - model: the trained model we want to attack
    - dataloader: data set for attack
    - attack_name: name of implemented attack
    - epsilons: how much we can change the initial picture
    - steps: list of numbers of queries
    - plot: plot picture or not
    - save: save accuracy array or not
    - aug_type: how to name plot and/or array
    - device: cpu or cuda
    
    Return:
    - accuracy under attack
    
    """
    
    model.to(device)
    
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    print('Clear accuracy on images without adversarial attack', 
          fb.utils.accuracy(fmodel, images, labels))
    
    if attack_name == 'Boundary':
        norms = []
        for step in steps:
            attack = fb.attacks.BoundaryAttack(init_attack=None, steps=step, spherical_step=0.01, 
                                               source_step=0.01, source_step_convergance=1e-07, 
                                               step_adaptation=1.5, tensorboard=False,
                                               update_stats_every_k=10)
            
            perturbed, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
            robust_accuracy = 1 - success.cpu().numpy().mean(axis=-1)
            
            q = torch.clip(perturbed[0], 0, 1)
            a = images - q
            a = a.view(-1)
            a = (a @ a).mean() / 32 / 32 / 3
            
            norms.append(float(a.cpu().numpy()))
    else:
        print('WRONG NAME, RESTART')
        return False
    
    if plot:
        # print(torch.linalg.norm(images-perturbed, ord=2))
        plt.figure(figsize=(15, 7)) # dpi=150
        plt.plot(steps, norms)
        
        plt.title('queries' + attack_name + '_' + aug_type)
        plt.xlabel('iterations')
        plt.ylabel('S')
        plt.grid()
        
        plt.savefig(attack_name + '_' + aug_type + '.png')
        plt.show()
    
    if save:
        np.save(attack_name + '_' + aug_type + '.npy', norms)
    
    model.to(torch.device('cpu'))
    
    return norms # robust_accuracy, advs, images, perturbed, 
