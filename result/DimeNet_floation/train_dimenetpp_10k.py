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

#train_src = "../../data/231213_train.lmdb"
#val_src = "../../data/231213_valid.lmdb"
all = 1
if all == 1:
    train_src = "../../data/feature_all_train.lmdb"
    val_src = "../../data/feature_all_valid.lmdb"
    epoch = 20
else:
    train_src = "../../data/feature_train.lmdb"
    val_src = "../../data/feature_valid.lmdb"
    epoch = 400

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
    'name': 'dimenetplusplus_KAN',
    'hidden_channels': 256, # if training is too slow for example purposes reduce the number of hidden channels
    'out_emb_channels': 192,
    'num_blocks': 3,
    'cutoff': 6.0,
    'num_radial': 6,
    'num_spherical': 7,
    'num_before_skip': 1,
    'num_after_skip': 2,
    'num_output_layers': 3,
    'regress_forces': False,
    'use_pbc': True,
    'strain_projection_channels': 32,
    'num_strain_layers': 4,
    'strain_final_dim': 16
#    'otf_graph': True
}

# Optimizer
optimizer = {
    'batch_size': 8,         # originally 32
    'eval_batch_size': 8,    # originally 32
    'num_workers': 8,
    'lr_initial': 0.001,
    'lr_gamma': 0.1,
    'lr_milestones':[20000, 40000, 60000],
    'warmup_steps': 10000,
    'warmup_factor': 0.2,
#    'optimizer': 'AdamW',
#    'optimizer_params': {"amsgrad": True},
#    'scheduler': "ReduceLROnPlateau",
#    'mode': "min",
#    'factor': 0.8,
#    'patience': 3,
    'max_epochs': epoch         # used for demonstration purposes
#    'ema_decay': 0.999,
#    'clip_grad_norm': 10,
#    'loss_energy': 'mae',
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
    model=copy.deepcopy(model), # copied for later use, not necessary in practice.
    dataset=dataset,
    optimizer=optimizer,
    identifier="IS2RE-dimenetpp-10k-lr1e-4",
    run_dir="./result", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=False, # if True, do not save checkpoint, logs, or results
    #is_vis=False,
    print_every=30,
    seed=2, # random seed to use
    logger="tensorboard", # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
)

#print(energy_trainer.model)

#energy_trainer.load_checkpoint(checkpoint_path='./checkpoint/dimenetpp_all.pt')
#print(energy_trainer.optimizer.state_dict()['param_groups'][0]['lr'])

#sd = energy_trainer.optimizer.state_dict()
#print(sd)
#sd['param_groups'][0]['lr'] = 0.0001
#energy_trainer.optimizer.load_state_dict(sd)
#print(energy_trainer.optimizer.state_dict()['param_groups'][0]['lr'])

#print(energy_trainer.optimizer.state_dict()['param_groups'][0]['lr'] == 2.0e-3)
#energy_trainer.optimizer = optimizer
#print(energy_trainer.optimizer)

energy_trainer.train()
