import argparse
import pytorch_lightning as pl
import torchvision
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

import models
import datasets

from torchmetrics.functional import accuracy
from attack import *

def extract_prefix(prefix, weights):
    from collections import OrderedDict
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix):]] = weights[key]
    return result 


class LightningWrapper(pl.LightningModule):
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset', type=str, help='Train dataset name', default='CIFAR10')
        parser.add_argument('--model', type=str, help='Model name to train', default='madry_cifar10')
        parser.add_argument('--batch', type=int, help='Batch size', default=128)
        parser.add_argument('--workers', type=int, help='Number of workers', default=4)
        parser.add_argument('--lr', type=float, help='Learning rate', default=0.05)
        parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
        parser.add_argument('--optim', type=str, help='Optimizer', choices=['SGD', 'Adam'], default='SGD')
        parser.add_argument('--scheduler', type=str, help='Scheduler', choices=['OneCycleLR', 'None'], default='OneCycleLR')
        
        parser.add_argument('--attack-bound-type', type=str, choices=['inf', 'l1', 'l2'], default='inf')
        parser.add_argument('--attack-eps', type=float, default=8.0/255.)
        parser.add_argument('--pgd-step-size', type=float, default=2.0/255.0)
        parser.add_argument('--pgd-num-steps', type=int, default=10)
        parser.add_argument('--ods-num-steps', type=int, default=2)
        parser.add_argument('--ods-step-size', type=float, default=8.0/255.0)
        
        parser.add_argument('--train-aug', type=str, choices=['no', 'gauss-noise', 'pgd', 'pgdods', 'ods'], default='no')
        parser.add_argument('--test-aug', type=str, choices=['no', 'gauss-noise', 'pgd', 'pgdods', 'ods'], default='no')
        return parser
    
    def create_attack(self, attack_name):
        if attack_name == 'no':
            return NoAttack(self.model, self.bound)
        elif attack_name == 'gauss-noise':
            return RandomAttack(self.model, self.bound, noise='gaussian', scale=self.hparams.attack_eps/2.0)
        elif attack_name == 'pgd':
            return PGDAttack(self.model, self.bound, max_steps=self.hparams.pgd_num_steps)
        elif attack_name == 'pgdods':
            return ODSPGDAttack(n_ods_steps=self.hparams.ods_num_steps, ods_step_size=self.hparams.ods_step_size,
                                model=self.model, bound=self.bound, max_steps=self.hparams.pgd_num_steps)
        else:
            raise NotImplemented(str(attack_name))
        
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters(args)
        
        self.model = models.get_model(self.hparams.model)
        self.train_dataset, self.val_dataset = datasets.get_dataset(self.hparams.dataset)
        self.bound = {'inf': LInfBounded, 'l1': L1Bounded, 'l2': L2Bounded}[self.hparams.attack_bound_type](eps=self.hparams.attack_eps,
                                                                                                                  step_size=self.hparams.pgd_step_size)
        self.train_attack = self.create_attack(self.hparams.train_aug)
        self.test_attack  = self.create_attack(self.hparams.test_aug) 


    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
    
        with EvalContext(self.model):
            with torch.no_grad():
                logits_nat = self(x)
                loss_nat = F.nll_loss(logits_nat, y)
        
        x_adv, success, dists = self.train_attack.attack(x, y)
        
        logits = self(x_adv)
        loss = F.nll_loss(logits, y)
        
        self.log("Loss/train_loss_adv", loss)
        self.log("Metrics/train_acc_adv", (~success).float().mean())
        self.log("Loss/train_loss_nat", loss_nat)
        self.log("Metrics/train_acc_nat", accuracy(logits_nat.argmax(dim=1), y))
        
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits_nat = self(x)
        loss_nat = F.nll_loss(logits_nat, y)
        preds_nat = torch.argmax(logits_nat, dim=1)
        acc_nat = accuracy(preds_nat, y)
        
        x_adv, success, dists = self.test_attack.attack(x,y)

        if stage:
            self.log(f"Loss/{stage}_loss_nat", loss_nat, prog_bar=True)
            self.log(f"Metrics/{stage}_acc_nat", acc_nat, prog_bar=True)
            self.log(f"Metrics/{stage}_acc_adv", (~success).float().mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        if self.hparams.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif self.hparams.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            
        steps_per_epoch = 50000 // self.hparams.batch
        
        opt_config = {"optimizer": optimizer}
        if self.hparams.scheduler != 'None':
            opt_config["lr_scheduler"] = {
                "scheduler": OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=steps_per_epoch,
                ),
                "interval": "step",
            }
        return opt_config#{"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.hparams.batch, 
                                           num_workers=self.hparams.workers,
                                           pin_memory=True, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.hparams.batch, 
                                           num_workers=self.hparams.workers,
                                           pin_memory=True)
