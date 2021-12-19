import argparse
from datetime import datetime
import random
import os
import sys
import numpy
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

sys.path.append('.')

from extensions.progressbar import LiteProgressBar
from extensions.custom_callbacks import *

from pl_module import LightningWrapper


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distrib', action='store_true', help='Multi-GPU distributed training')
    parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int, help='Distributed training world size')
    parser.add_argument('--test', action='store_true')
    return parser

EXPERIMENTS_PATH = 'experiments/'

def main():
    experiment_name = sys.argv[1:]
    for index in range(len(experiment_name)):
        while experiment_name[index][0] == '-':
            experiment_name[index] = experiment_name[index][1:]
    experiment_name = '_'.join(experiment_name)
    experiment_name = experiment_name.replace('/','-').replace(' ','').replace(',','').replace('.','')

    pl.seed_everything(42)
    parser = arg_parser()
    parser = LightningWrapper.add_model_specific_args(parser)
    args = parser.parse_args()
    
    datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_folder = os.path.join(EXPERIMENTS_PATH, experiment_name, datetime_str)
    os.makedirs(experiment_folder)
    
    kwargs = {}
    if args.distrib:
        kwargs['accelerator'] = 'ddp'
        kwargs['sync_batchnorm'] = True
    
    wrapper = LightningWrapper(args)
    trainer = pl.Trainer(
        gpus=1 if not args.distrib else args.gpus, 
        max_epochs=args.epochs, 
        num_sanity_val_steps=0, 
        default_root_dir=None,
        deterministic=False,
        logger=None,
#        logger=[
#             pl_loggers.TensorBoardLogger(TENSORBOARD_PATH, name=args.experiment_name, version=datetime_str),
#             pl_loggers.TensorBoardLogger('/tensorboard')
#        ],
        enable_model_summary=False,
        callbacks=[
            LiteProgressBar(), 
            CodeSnapshotter(experiment_folder),
            EnvironmentCollector(experiment_folder),
            MetricLogger(experiment_folder),
            ParamsLogger(experiment_folder),
            ModelCheckpoint(dirpath=os.path.join(experiment_folder, 'weights'), filename='{epoch}-{step}', save_top_k=-1),
        ], 
        **kwargs
    )

    trainer.fit(wrapper)
    

if __name__ == '__main__':
    main()
