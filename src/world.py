'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import os
from os.path import join
from enum import Enum
from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

config = {}
config['batch_size'] = args.batch_size
config['lr'] = args.lr
config['dataset'] = args.dataset
config['layers'] = args.layers
config['emb_dim'] = args.emb_dim
config['model'] = args.model
config['decay'] = args.decay
config['epochs'] = args.epochs
config['top_k'] = args.top_k
config['verbose'] = args.verbose
config['epochs_per_eval'] = args.epochs_per_eval
config['seed'] = args.seed
config['loadedModel'] = args.loadedModel
config['test_ratio'] = args.test_ratio
config['u_sim'] = args.u_sim
config['i_sim'] = args.i_sim
config['edge'] = args.edge
config['i_top_k'] = args.i_top_k
config['u_top_k'] = args.u_top_k
config['self_sim'] = bool(args.self_sim)
config['weight_mode'] = args.weight_mode
config['shuffle'] = args.shuffle
config['base'] = args.base
config['refresh'] = args.refresh
config['full_sample'] = args.full_sample
config['neg_samples'] = args.neg_samples
