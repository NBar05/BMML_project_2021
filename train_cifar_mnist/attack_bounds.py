import torch
import numpy as np

class Bounded:
    def __init__(self, eps=None, step_size=None):
        self.eps = eps
        self.step_size = step_size
        
    def project(self, x_orig, x_new):
        raise NotImplementedError()
    
    def step(self, x, grad):
        raise NotImplementedError()
        
    def perturb(self, x):
        raise NotImplementedError()
        

class LInfBounded(Bounded):
    def project(self, x_orig, x_new):
        return torch.clamp(x_orig + torch.clamp(x_new - x_orig, min=-self.eps, max=self.eps), min=0, max=1)
    
    def step(self, x, grad):
        return grad.sign() * self.step_size + x
    
    def perturb(self, x):
        return torch.clamp(x + self.eps * 2 * (torch.rand_like(x) - 0.5), min=0, max=1)
    
    def dist(self, x_orig, x_new, dim=None):
        return torch.amax((x_orig-x_new).abs(), dim=dim)
    
class LPBounded(Bounded):
    def __init__(self, p=2, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.p = p
        
    def project(self, x_orig, x_new):
        return torch.clamp(x_orig + (x_new - x_orig).renorm(p=self.p, dim=0, maxnorm=self.eps), min=0, max=1)
    
    def step(self, x, grad):
        grad_norm = grad.norm(p=self.p, dim=(1,2,3), keepdim=True)+1e-9
        return x + self.step_size * grad / grad_norm
    
    def perturb(self, x):
        noise = torch.randn_like(x)
        noise_norm = noise.norm(p=self.p, dim=(1,2,3), keepdim=True)+1e-9
        return torch.clamp(x + self.eps * noise / noise_norm, min=0, max=1)
    
    def dist(self, x_orig, x_new, dim=None):
        return (x_orig-x_new).norm(p=self.p, dim=dim)
    
class L2Bounded(LPBounded):
    def __init__(self, *args, **kwargs):
        super().__init__(p=2, *args, **kwargs)

class L1Bounded(LPBounded):
    def __init__(self, *args, **kwargs):
        super().__init__(p=1, *args, **kwargs)