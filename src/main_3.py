'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import torch.backends
import torch.mps
import numpy as np
from procedure import run_experiment_2
from utils import print_metrics, set_seed
import data_prep as dp 
from world import config

# STEP 1: set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

# STEP 2: Load the data
train_df, test_df = dp.load_data_from_adj_list(dataset = config['dataset'])

num_users = train_df['user_id'].nunique()
num_items = train_df['item_id'].nunique()
num_interactions = len(train_df)

stats = {'num_users': num_users, 'num_items': num_items,  'num_interactions': num_interactions}

#seeds = [2020, 12, 89, 91, 41]
seeds = [2020]

all_bi_metrics = []
all_bi_losses = []

recalls = []
precs = []
f1s = []
ncdg = []
max_indices = []
exp_n = 1

for seed in seeds:
    
    set_seed(seed)
    
    losses, metrics = run_experiment_2(o_train_df = train_df, o_test_df=test_df, g_seed = seed, exp_n = exp_n, device=device, verbose=config['verbose'])
    
    ncdg.append(np.max(metrics['ncdg']))
    max_idx = np.argmax(metrics['ncdg'])
    recalls.append(metrics['recall'][max_idx])
    precs.append(metrics['precision'][max_idx])
    f1s.append(metrics['f1'][max_idx])
    
    max_indices.append(max_idx)
    
    all_bi_losses.append(losses)
    all_bi_metrics.append(metrics)
    
    exp_n += 1

print_metrics(recalls, precs, f1s, ncdg, max(max_indices), stats=stats)

