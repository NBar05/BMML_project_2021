import copy
import numpy as np
import torch

from torch import nn
from torch import linalg
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def ods_attack(model: nn.Module, inputs: torch.Tensor, nu: float=0.02, num_of_steps: int=2, device=torch.device('cpu')):
    """
    Implement ODS change of the input data
    
    """
    
    # define distribution for attack
    distribution = torch.distributions.uniform.Uniform(-1, 1)
    
    # copy model for ods (means without grad flow)
    model.to(torch.device('cpu'))
    
    model_for_ods = copy.deepcopy(model)
    for param in model_for_ods.parameters():
        param.requires_grad = False

    model_for_ods.to(device)
    model.to(device)
    
    # and copy data
    inputs_for_ods = inputs.clone()
    
    # make ODS steps
    for _ in range(num_of_steps):
        # turn on gradient for input data
        inputs_for_ods = nn.Parameter(inputs_for_ods.detach().clone(), requires_grad=True)

        # get logits
        outputs_for_ods = model_for_ods(inputs_for_ods)
        
        # get random weights for dot with logits
        with torch.no_grad():
            weights = distribution.sample(sample_shape=outputs_for_ods.size()).to(device)

        # get grads
        vector_for_back = (outputs_for_ods * weights).sum(axis=1).sum()
        vector_for_back.backward()
        
        # make ods vector
        vector_ods = inputs_for_ods.grad / linalg.norm(inputs_for_ods.grad.flatten(start_dim=1, end_dim=-1), 
                                                       dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        
        # change input data according to grad directions
        inputs_for_ods.data = torch.clip(inputs_for_ods.data + nu * vector_ods, min=0.0, max=1.0) # torch.sign(vector_ods)
        
        # zero grad for clean next steps
        inputs_for_ods.grad.zero_()
    
    return inputs_for_ods.data
