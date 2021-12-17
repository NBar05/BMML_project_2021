import time
import copy
import numpy as np

import torch

from torch import nn
from torch import linalg
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms, datasets, models

from utils_ods_attack import ods_attack


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, device=torch.device('cpu'),
          augment: bool=True, 
          noise_sd: float=0.1, 
          attack: bool=True, nu: float=0.02, num_of_steps: int=2):
    
    # switch to train mode
    model.train()
    
    batch_time = []
    losses, accs = [], []
    
    for i, (inputs, targets) in enumerate(loader, start=1):        
        start = time.time()
        
        optimizer.zero_grad()
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # change initial data or not
        if augment:
            # augment inputs with ODS shifts or noise
            if attack:
                inputs = ods_attack(model, inputs, nu, num_of_steps, device=device)
            else:
                inputs = torch.clip(inputs + torch.randn_like(inputs, device=device) * noise_sd, min=0.0, max=1.0)
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # record loss
        losses.append(loss.cpu().data)
        # and accuracy
        accs.append((outputs.argmax(dim=1) == targets).to(torch.float).mean().cpu().detach())
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        end = time.time()
        
        batch_time.append(end - start)

        if i % 150 == 0:
            print(
                f"Train (after {i} batches): "\
                f"Current time: {batch_time[-1]:.3f} (mean: {np.mean(batch_time):.3f}) "\
                f"Current loss: {losses[-1]:.3f} (mean: {np.mean(losses):.3f}) "\
                f"Current accuracy: {accs[-1]:.3f} (mean: {np.mean(accs):.3f}) "
            )
            
    return (np.mean(losses), np.mean(accs))

def test(loader: DataLoader, model: torch.nn.Module, criterion, device=torch.device('cpu'),
         augment: bool=True, 
         noise_sd: float=0.1, 
         attack: bool=True, nu: float=0.02, num_of_steps: int=2):

    # switch to eval mode
    model.eval()
    
    batch_time = []
    losses, accs = [], []

    for i, (inputs, targets) in enumerate(loader, start=1):
        start = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        
        # change initial data or not
        if augment:
            # augment inputs with ODS shifts or noise
            if attack:
                inputs = ods_attack(model, inputs, nu, num_of_steps, device=device)
            else:
                inputs = torch.clip(inputs + torch.randn_like(inputs, device=device) * noise_sd, min=0.0, max=1.0)
        
        with torch.no_grad():
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # record loss
            losses.append(loss.cpu().data)
            # and accuracy
            accs.append((outputs.argmax(dim=1) == targets).to(torch.float).mean().cpu().detach())

            # measure elapsed time
            end = time.time()

            batch_time.append(end - start)

            if i % 40 == 0:
                print(
                    f"Validation (after {i} batches): "\
                    f"Current time: {batch_time[-1]:.3f} (mean: {np.mean(batch_time):.3f}) "\
                    f"Current loss: {losses[-1]:.3f} (mean: {np.mean(losses):.3f}) "\
                    f"Current accuracy: {accs[-1]:.3f} (mean: {np.mean(accs):.3f}) "
                )

    return (np.mean(losses), np.mean(accs))

def short_test(loader: DataLoader, model: torch.nn.Module, device=torch.device('cpu')):
    
    model.to(device)
    model.eval()
    
    accs = []
    
    with torch.no_grad():
        for (inputs, targets) in loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            accs.append((outputs.argmax(dim=1) == targets).to(torch.float).cpu().detach())
            
    model.to(torch.device('cpu'))
    
    return (torch.cat(accs).mean().numpy() * 100).round(3)
