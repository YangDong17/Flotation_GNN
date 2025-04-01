from ocpmodels.trainers import EnergyTrainer
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging
setup_logging()

import numpy as np
import copy
import os
import torch
#train_src = "../data/IS2RE_231031_train_no.lmdb"
#val_src = "../data/IS2RE_231031_valid.lmdb"

#train_src = "../data/231026_train_no.lmdb"
#val_src = "../data/231026_valid.lmdb"
#train_src = "../data/231213_train.lmdb"
#val_src = "../data/231213_valid.lmdb"

train_src = "../data/231213_train_all.lmdb"
val_src = "../data/231213_valid_all.lmdb"
train_dataset = SinglePointLmdbDataset({"src": train_src})
#train_dataset = LmdbDataset({"src": train_src})
print(len(train_dataset))
energies = []
#print(train_dataset)
#for data in train_dataset:
#  energies.append(data.y_relaxed)
#print(train_dataset[0])
#print(train_dataset[-1])
energies = [data.y_relaxed for data in train_dataset]
#print(energies)


mean = np.mean(energies)
stdev = np.std(energies)

print(mean)
print(stdev)
# Task
task = {
  "dataset": "single_point_lmdb",
  "description": "Relaxed state energy prediction from initial structure.",
  "type": "regression",
  "metric": "mae",
  "labels": ["relaxed energy"],
}
# Model
model = {
    'name': 'painn',
    'hidden_channels': 1024,
    'num_layers': 6,
    'num_rbf': 128,
    'cutoff': 12.0,
    'max_neighbors': 50,
    'scale_file': './painn_nb6_scaling_factors.pt',
    'regress_forces': False,
    'use_pbc': True,
}

# Optimizer
#optimizer = {
#    'batch_size': 8,         # originally 32
#    'eval_batch_size': 8,    # originally 32
#    'num_workers': 8,
#    'lr_initial': 0.0001,
#    'lr_gamma': 0.1,
#    'lr_milestones':115082,
#    'warmup_steps': 57541,
#    'warmup_factor': 0.2,
#    'optimizer': 'AdamW',
#    'optimizer_params': {"amsgrad": True},
#    'scheduler': "ReduceLROnPlateau",
#    'mode': "min",
#    'factor': 0.8,
#    'patience': 3,
#    'max_epochs': 900,         # used for demonstration purposes
#    'ema_decay': 0.999,
#    'clip_grad_norm': 10,
#    'loss_energy': 'mae',
#}
optimizer = {
  'batch_size': 8,
  'eval_batch_size': 8,
  'load_balancing': 'atoms',
  'num_workers': 2,
  'optimizer': 'AdamW',
  'optimizer_params': {"amsgrad": True},
  'lr_initial': 1.e-6,
  'scheduler': 'ReduceLROnPlateau',
  'mode': 'min',
  'factor': 0.8,
  'patience': 3,
  'max_epochs': 30,
  'energy_coefficient': 1,
  'ema_decay': 0.999,
  'clip_grad_norm': 10,
  'loss_energy': 'mae',
  'weight_decay': 2e-6  # 2e-6 (TF weight decay) / 1e-4 (lr) = 2e-2    
}
# Dataset
dataset = [
  {'src': train_src,
   'normalize_labels': True,
   'target_mean': mean,
   'target_std': stdev,
  }, # train set
  {'src': val_src}, # val set (optional)
]

energy_trainer = EnergyTrainer(
    task=task,
    model=copy.deepcopy(model),
    dataset=dataset,
    optimizer=optimizer,
    identifier="IS2RE-painn-init",
    run_dir="./result", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=False, # if True, do not save checkpoint, logs, or results
    #is_vis=False,
    print_every=20,
    seed=2, # random seed to use
    logger="tensorboard", # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
)

#energy_trainer.load_checkpoint(checkpoint_path='../checkpoints/painn_h1024_bs4x8_is2re_all.pt')

#sd = energy_trainer.optimizer.state_dict()
#print(sd['param_groups'])
#sd['param_groups'][0]['lr'] = 0.00001
#energy_trainer.optimizer.load_state_dict(sd)

#energy_trainer.step=len(energy_trainer.train_loader)
#print(len(energy_trainer.train_loader))
#print(energy_trainer.step)
#print(energy_trainer.step/len(energy_trainer.train_loader))

energy_trainer.train()
print("###########################################################")

