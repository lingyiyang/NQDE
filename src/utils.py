import math
import itertools
import numpy as np
import torch
import torchcde
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
import yaml
from torchviz import make_dot

import random
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

import sys
sys.path.append('../')


def get_data(num_timepoints=100, datasize=128):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)

    start = torch.rand(datasize) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:int(datasize/2)] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    X = torch.stack([t.unsqueeze(0).repeat(datasize, 1), x_pos, y_pos], dim=2)
    #y = torch.zeros(datasize, dtype=int)
    y = torch.zeros(datasize, dtype=int)
    y[:int(datasize/2)] = 1

    perm = torch.randperm(datasize)
    X = X[perm]
    y = y[perm]

    return X, y


def save_outputs(configs, model, losses, proportion_correct, nfe_forward, nfe_backward):
    today = datetime.now()   # Get date
    datestring = today.strftime("%Y_%m_%d_%H_%M")

    output_dir = Path(__file__).resolve().parent.parent /'results'/ datestring
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)   # Create folder

    results = {
        'losses':losses,
        'nfe_fs':nfe_forward,
        'nfe_bs': nfe_backward,
        'test_accuracy': proportion_correct}

    with open(output_dir/'configs_used.yaml', 'w') as file:
        yaml.dump(configs | results, file)

    torch.save(model, output_dir/'model.pt')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def modify_dict_hyperparameters(configs):
    '''
    Deconstruct configs into a list of configs if key values contain lists in 
    the original config. This is for the purpose of hyperparameter 
    optimisation
    
    Parameters  configs: a dictionary of the config parameters, with lists in 
                         some key values if multiple values should be tried 
                         for hyperparameter optimisation
    Returns     modified: a list of configurations, with cartesian products on
                          the keys where values are lists    
    '''
    
    modified = []
    
    var = {k: i for k, i in configs.items() if isinstance(i, list)}
    fixed = {k: i for k, i in configs.items() if k not in var}      

    if bool(var):
        keys_var, values_var = zip(*var.items())
        new = [dict(zip(keys_var, v)) for v in itertools.product(*values_var)]

        modified = [{**fixed, **i} for i in new]
    else:
        modified.append(configs)
                           
    return modified



