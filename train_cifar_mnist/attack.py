from attack_bounds import *
import torch
import numpy as np



class EvalContext:
    def __init__(self, model):
        self.model = model
        self.model_state = model.training
        
    def __enter__(self):
        self.model_state = self.model.training
        self.model.eval()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.train(self.model_state)
        

def margin_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(logits.shape[1])[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss


class PGDAttack:
    def __init__(self, model, bound: Bounded, max_steps=100, loss_func: str = 'ce'):
        super().__init__()
        self.model = model
        self.bound = bound
        self.max_steps = max_steps
        self.loss_func = loss_func
        
        self._t = 0
        
    def eval_loss(self, logits, y_true, y_desired):
        return margin_loss(logits, y_true)
    
    def step_size(self):
        return self.bound.step_size
        
    def attack(self, X, y_true, y_desired=None):
        with EvalContext(self.model), torch.set_grad_enabled(True):
            X_orig = X.clone()
            X_adv = self.bound.perturb(X_orig.detach().clone()).requires_grad_(True)
            opt = torch.optim.SGD([X_adv], lr=self.bound.step_size)

            for i in range(self.max_steps):
                opt.zero_grad()
                logits = self.model(X_adv)

                loss = self.eval_loss(logits, y_true, y_desired)
                loss.backward()
                self.step_size()
                X_adv.data = self.bound.step(X_adv.detach(), X_adv.grad)
                X_adv.data = self.bound.project(X_orig, X_adv.detach())

                self._t += 1

            with torch.no_grad():
                X_adv = X_adv.detach()
                success = self.model(X_adv).argmax(dim=1) != y_true

                b = X_orig.shape[0]
                dists = self.bound.dist(X_orig.view(b,-1), X_adv.view(b,-1), dim=1)
                
        return X_adv, success, dists
    

class ODSPGDAttack(PGDAttack):
    def __init__(self, n_ods_steps=2, ods_step_size=8.0/255., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_ods_steps = n_ods_steps
        self.ods_step_size = ods_step_size
        self.pgd_step_size = self.bound.step_size
        
    def eval_loss(self, logits, y_true, y_desired):
        if self._t < self.n_ods_steps:
            #print('ods loss')
            self.w_d = torch.rand_like(logits) * 2 - 1
            return (self.w_d * logits).sum()
        else:
            #print('usual loss')
            return margin_loss(logits, y_true)
        
    def step_size(self):
        self.bound.step_size = self.ods_step_size if self._t < self.n_ods_steps else self.pgd_step_size
        return self.bound.step_size
    
class RandomAttack:
    def __init__(self, model, bound: Bounded, noise='uniform', scale=1.0):
        super().__init__()
        self.model = model
        self.bound = bound
        self.noise = noise
        self.scale = scale
        
        if  noise == 'uniform':
            self.distribution = torch.distributions.Uniform(-scale, scale)
        else:
            self.distribution = torch.distributions.Normal(loc=0, scale=scale)
        
    def attack(self, X, y_true):
        with torch.no_grad():
            with EvalContext(self.model):
                X_adv = self.bound.project(X, X + self.distribution.sample(sample_shape=X.shape))
            
                success = self.model(X_adv) != y_true
                batch = X.shape[0]
                dists = self.bound.dist(X.view(batch, -1), X_adv.view(batch, -1))
        
        return X_adv, success, dists
    
    
class NoAttack:
    def __init__(self, model, bound: Bounded):
        super().__init__()
        self.model = model
        
    def attack(self, X, y_true, y_desired=None):
        X_adv = X
        with torch.no_grad():
            with EvalContext(self.model):
                logits = self.model(X)
                success = logits.argmax(dim=1) != y_true
        
        return X_adv, success, torch.zeros(X_adv.shape[0]).to(X_adv.device)
    
    
class RepeatedAttack:
    def __init__(self, attack, n_restarts=10, *args, **kwargs):
        super().__init__()
        self.base_attack = attack
        self.n_restarts = n_restarts
        
    def attack(self, X, *args, **kwargs):
        X_adv   = X.clone()
        success = torch.zeros(X_adv.shape[0], dtype=torch.bool).to(X_adv.device)
        dists   = torch.ones(X_adv.shape[0], dtype=torch.float32).to(X_adv.device) * np.inf
        
        for i in range(self.n_restarts):
            X_adv_cur, success_cur, dists_cur = self.base_attack.attack(X, *args, **kwargs)
            #print(dists.shape, dists_cur.shape)
            mask_update = (success_cur & (~success)) | ((success & success_cur) & (dists_cur < dists))
            #print(mask_update)
            X_adv[mask_update] = X_adv_cur[mask_update]
            success[mask_update] = success_cur[mask_update]
            dists[mask_update] = dists_cur[mask_update]
            
        return X_adv, success, dists