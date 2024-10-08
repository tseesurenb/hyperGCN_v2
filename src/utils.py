'''
Created on Sep 1, 2024
Pytorch Implementation of hyperGCN: Hyper Graph Convolutional Networks for Collaborative Filtering
'''

import torch
import random
import matplotlib
matplotlib.use('Agg')

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from model import RecSysGNN
from sklearn import preprocessing as pp
from world import config
import data_prep as dp
from data_prep import get_edge_index, create_uuii_adjmat
import time
import sys

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def print_metrics(recalls, precs, f1s, ncdg, max_indices, stats): 
    
    print(f" Dataset: {config['dataset']}, num_users: {stats['num_users']}, num_items: {stats['num_items']}, num_interactions: {stats['num_interactions']}")
    
    if config['edge'] == 'bi':
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs} | Shuffle: {br}{config['shuffle']}{rs} | Test Ratio: {br}{config['test_ratio']}{rs} | Base: {br}{config['base']}{rs}")
    else:
        print(f"   MODEL: {br}{config['model']}{rs} | EDGE TYPE: {br}{config['edge']}{rs} | #LAYERS: {br}{config['layers']}{rs} | SIM (mode-{config['weight_mode']}, self-{config['self_sim']}): {br}u-{config['u_sim']}(topK {config['u_top_k']}), i-{config['i_sim']}(topK {config['i_top_k']}){rs} | BATCH_SIZE: {br}{config['batch_size']}{rs} | DECAY: {br}{config['decay']}{rs} | EPOCHS: {br}{config['epochs']}{rs} | Shuffle: {br}{config['shuffle']}{rs} | Test Ratio: {br}{config['test_ratio']}{rs}")

    metrics = [("Recall", recalls), 
           ("Prec", precs), 
           ("F1 score", f1s), 
           ("NDCG", ncdg)]

    for name, metric in metrics:
        values_str = ', '.join([f"{x:.4f}" for x in metric[:5]])
        mean_str = f"{round(np.mean(metric), 4):.4f}"
        std_str = f"{round(np.std(metric), 4):.4f}"
        
        # Apply formatting with bb and rs if necessary
        if name in ["F1 score", "NDCG"]:
            mean_str = f"{bb}{mean_str}{rs}"
        
        print(f"{name:>8}: {values_str} | {mean_str}, {std_str}")
    
    print(f"{35*'-'}")    
    print(f"   Max NDCG occurs at epoch {br}{(max_indices) * config['epochs_per_eval']}{rs}")

def get_metrics(user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_df, K, device, batch_size=100):
    
    _debug = False
    
    # Ensure embeddings are on the correct device
    user_Embed_wts = user_Embed_wts.to(device)
    item_Embed_wts = item_Embed_wts.to(device)

    assert n_users == user_Embed_wts.shape[0]
    assert n_items == item_Embed_wts.shape[0]

    # Initialize metrics
    total_recall = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    num_batches = (n_users + batch_size - 1) // batch_size

    # Prepare interaction tensor for the batch
    i = torch.stack((
        torch.LongTensor(train_df['user_id'].values),
        torch.LongTensor(train_df['item_id'].values)
    )).to(device)
    
    v = torch.ones(len(train_df), dtype=torch.float32).to(device)
    interactions_t = torch.sparse_coo_tensor(i, v, (n_users, n_items), device=device).to_dense()
    
    if _debug:
        print_tensor = interactions_t
        print(f"\n interactions_t ({print_tensor.shape})\n: {print_tensor}\n")


    # Collect results across batches
    all_topk_relevance_indices = []
    all_user_ids = []

    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_user_indices = torch.arange(batch_start, batch_end).to(device)

        # Extract embeddings for the current batch
        user_Embed_wts_batch = user_Embed_wts[batch_user_indices]
        #relevance_score_batch = torch.matmul(user_Embed_wts_batch, item_Embed_wts.t())
        relevance_score_batch = torch.matmul(user_Embed_wts_batch, torch.transpose(item_Embed_wts,0, 1))

        # Mask out training user-item interactions from metric computation
        relevance_score_batch = relevance_score_batch * (1 - interactions_t[batch_user_indices])

        # Compute top scoring items for each user
        topk_relevance_indices = torch.topk(relevance_score_batch, K).indices
        all_topk_relevance_indices.append(topk_relevance_indices)
        all_user_ids.extend(batch_user_indices.cpu().numpy())

    # Combine results
    topk_relevance_indices = torch.cat(all_topk_relevance_indices).cpu().numpy()
    
    if _debug:
        print_tensor = topk_relevance_indices
        print(f"\n topk_relevance_indices ({print_tensor.shape})\n: {print_tensor}\n")

    # Measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = test_df.groupby('user_id')['item_id'].apply(list).reset_index()
    
    if _debug:
        print_tensor = test_interacted_items
        print(f"\n test_interacted_items ({print_tensor.shape})\n: {print_tensor}\n")
   
    # Merge test interactions with top-K predicted relevance indices
    metrics_df = pd.merge(test_interacted_items, pd.DataFrame({'user_id': all_user_ids, 'top_rlvnt_itm': topk_relevance_indices.tolist()}), how='left', on='user_id')
    
    if _debug:
        print_tensor = metrics_df
        print(f"\n metrics_df ({print_tensor.shape})\n: {print_tensor}\n")
    
    # Handle missing values and ensure that item_id and top_rlvnt_itm are lists
    metrics_df['item_id'] = metrics_df['item_id'].apply(lambda x: x if isinstance(x, list) else [])
    metrics_df['top_rlvnt_itm'] = metrics_df['top_rlvnt_itm'].apply(lambda x: x if isinstance(x, list) else [])

    # Calculate intersection items
    metrics_df['intrsctn_itm'] = [list(set(a).intersection(b)) for a, b in zip(metrics_df.item_id, metrics_df.top_rlvnt_itm)]
    
    if _debug:
        print_tensor = metrics_df
        print(f"\n metrics_df ({print_tensor.shape})\n: {print_tensor}\n")

    # Calculate recall, precision, and nDCG
    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / len(x['item_id']) if len(x['item_id']) > 0 else 0, axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intrsctn_itm']) / K, axis=1)
    
    if _debug:
        print_tensor = metrics_df
        print(f"\n metrics_df ({print_tensor.shape})\n: {print_tensor}\n")


    # Generate a binary relevance matrix for test interactions (same as in the first function)
    test_matrix = np.zeros((len(metrics_df), K))
    for i, row in metrics_df.iterrows():
        relevant_items = set(row['item_id'])
        predicted_items = row['top_rlvnt_itm']
        length = min(K, len(relevant_items))
        test_matrix[i, :length] = 1
    
    
    if _debug:    
        print_tensor = test_matrix
        print(f"\n test_matrix ({print_tensor.shape})\n: {print_tensor}\n")


    # Compute IDCG (Ideal DCG)
    idcg = np.sum(test_matrix * 1./np.log2(np.arange(2, K + 2)), axis=1)
    
    if _debug:
        print_tensor = idcg
        print(f"\n idcg ({print_tensor.shape})\n: {print_tensor}\n")
    
    # Compute DCG based on predicted relevance
    dcg_matrix = np.zeros((len(metrics_df), K))
    for i, row in metrics_df.iterrows():
        relevant_items = set(row['item_id'])
        predicted_items = row['top_rlvnt_itm']
        dcg_matrix[i] = [1 if item in relevant_items else 0 for item in predicted_items]
    
    dcg = np.sum(dcg_matrix * (1. / np.log2(np.arange(2, K + 2))), axis=1)
    
    if _debug:
        print_tensor = dcg
        print(f"\n dcg ({print_tensor.shape})\n: {print_tensor}\n")

    # Handle cases where idcg == 0 to avoid division by zero
    idcg[idcg == 0.] = 1.

    # Compute nDCG as DCG / IDCG
    ndcg = dcg / idcg
    
    if _debug:
        print_tensor = ndcg
        print(f"\n ndcg ({print_tensor.shape})\n: {print_tensor}\n")
        sys.exit()

    # Set NaNs in nDCG to zero
    ndcg[np.isnan(ndcg)] = 0.

    # Aggregate metrics
    total_recall = metrics_df['recall'].mean()
    total_precision = metrics_df['precision'].mean()
    total_ndcg = np.mean(ndcg)
    
    return total_recall, total_precision, total_ndcg

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def minibatch(*tensors, batch_size):

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def make_adj_list(data, all_items):
    all_items_set = set(all_items)

    # Group by user_id and aggregate item_ids into lists (positive items)
    pos_items = data.groupby('user_id')['item_id'].agg(list)
    
    # Compute neg_items by subtracting the pos_items from all_items for each user
    neg_items = pos_items.apply(lambda pos: list(all_items_set.difference(pos)))
    
    # Create a dictionary with user_id as the key and a sub-dictionary with both pos_items and neg_items
    full_adj_list_dict = {
        user_id: {'pos_items': pos_items[user_id], 'neg_items': neg_items[user_id]}
        for user_id in pos_items.index
    }

    # Clear unnecessary variables from memory
    del pos_items, neg_items, all_items_set
    
    return full_adj_list_dict


def make_adj_list_batched(data, all_items, neg_sample_size):
    all_items_set = set(all_items)

    # Group by user_id and aggregate item_ids into lists (positive items)
    pos_items = data.groupby('user_id')['item_id'].agg(list)

    # Compute neg_items by subtracting the pos_items from all_items for each user
    neg_items = pos_items.apply(lambda pos: list(all_items_set.difference(pos)))

    # Create a dictionary with user_id as the key and a sub-dictionary with pos_items and neg_item_batches
    full_adj_list_dict = {}
    j = 0
    for user_id in pos_items.index:
        pos_item_list = pos_items[user_id]
        neg_item_list = neg_items[user_id]

        # Divide neg_items into batches of size batch_size
        neg_item_batches = [neg_item_list[i:i + neg_sample_size] for i in range(0, len(neg_item_list), neg_sample_size)]
          
        # Add user_id, pos_items, and the neg_item_batches to the dictionary
        full_adj_list_dict[user_id] = {
            'pos_items': pos_item_list,
            'neg_batches': len(neg_item_batches),
            'neg_item_batches': neg_item_batches  # Store all batches in a list
        }

    #for i in range(10):
    #    print(f"Full Adj List Dict: {full_adj_list_dict[i]['neg_batches']}, {neg_sample_size}, {len(full_adj_list_dict[i]['neg_item_batches'][full_adj_list_dict[i]['neg_batches'] - 2])}")
        #print(f"Full Adj List Dict: {full_adj_list_dict[i]['pos_items']}")    
    #sys.exit()
    # Clear unnecessary variables from memory
    del pos_items, neg_items, all_items_set

    return full_adj_list_dict




def neg_uniform_sample(train_df, full_adj_list, n_usr):
    interactions = train_df.to_numpy()
    users = interactions[:, 0].astype(int)
    pos_items = interactions[:, 1].astype(int)
    
    #neg_items = []

    #for u in users:
    #    neg_list = full_adj_list[u]['neg_items']
    #    neg_items.append(neg_list[np.random.randint(0, len(neg_list))])
    
    neg_items = np.array([full_adj_list[u]['neg_items'][np.random.randint(0, len(full_adj_list[u]['neg_items']))] for u in users])
        
    pos_items = [item + n_usr for item in pos_items]
    neg_items = [item + n_usr for item in neg_items]
    
    S = np.column_stack((users, pos_items, neg_items))
    
    del users, pos_items, neg_items
    
    return S

def get_random_slice(items, N):
    """Gets a random slice of length N from the items list, ensuring it's within bounds."""
    start = np.random.randint(len(items) - N + 1)  # Adjust to start within bounds
    return items[start:start + N]

def multiple_neg_uniform_sample(train_df, full_adj_list, n_usr):
    interactions = train_df.to_numpy()
    users = interactions[:, 0].astype(int)
    pos_items = interactions[:, 1].astype(int)
    
    
    #For each user, generate N negative samples
    # neg_items_list = np.array([
    #     np.random.choice(full_adj_list[u]['neg_items'], size=N, replace=True) 
    #     for u in users
    # ])
    
    #neg_items_list = np.array([get_random_slice(full_adj_list[u]['neg_items'], N) for u in users])    
    
    neg_items_list = np.array([
    full_adj_list[u]['neg_item_batches'][random.randint(0, full_adj_list[u]['neg_batches'] - 2)]
    for u in users
    ])

    
    # Adjust positive and negative item indices by adding n_usr
    pos_items = [item + n_usr for item in pos_items]
    neg_items_list = [[item + n_usr for item in neg_list] for neg_list in neg_items_list]  # Keep the list structure
    
    # Stack the users, positive items, and the list of negative items
    S = np.column_stack((users, pos_items, neg_items_list))
    
    return S

def full_uniform_sample(train_df, full_adj_list, n_usr):
    users = np.random.randint(0, n_usr, len(train_df))
    
    # Pre-allocate arrays for positive and negative items
    pos_items = np.empty(len(train_df), dtype=np.int32)
    neg_items = np.empty(len(train_df), dtype=np.int32)
    
    # Vectorized sampling for positive and negative items
    for idx, u in enumerate(users):
        pos_list = full_adj_list[u]['pos_items']
        neg_list = full_adj_list[u]['neg_items']
        
        pos_items[idx] = pos_list[np.random.randint(0, len(pos_list))] + n_usr
        neg_items[idx] = neg_list[np.random.randint(0, len(neg_list))] + n_usr
    
    # Stack the columns into a single array
    S = np.column_stack((users, pos_items, neg_items))
    
    return S

def full_uniform_sample_naive(train_df, full_adj_list, n_usr):
     
    users = np.random.randint(0, n_usr, len(train_df))
    
    pos_items = []
    neg_items = []
    
    for u in users:
        pos_list = full_adj_list[u]['pos_items']
        neg_list = full_adj_list[u]['neg_items']
        pos_items.append(pos_list[np.random.randint(0, len(pos_list))] + n_usr)
        neg_items.append(neg_list[np.random.randint(0, len(neg_list))] + n_usr)
    
    
    S = np.column_stack((users, pos_items, neg_items))
    
    del users, pos_items, neg_items
    
    return S
                 
def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
                
def plot_results(plot_name, num_exp, epochs, all_bi_losses, all_bi_metrics, all_knn_losses, all_knn_metrics):
    plt.figure(figsize=(14, 5))  # Adjust figure size as needed
    
    num_test_epochs = len(all_bi_losses[0]['loss'])
    epoch_list = [(j + 1) for j in range(num_test_epochs)]
             
    for i in range(num_exp):
        
        plt.subplot(1, 3, 1)
        # BI Losses
        plt.plot(epoch_list, all_bi_losses[i]['loss'], label=f'Exp {i+1} - BI Total Training Loss', linestyle='-', color='blue')
        plt.plot(epoch_list, all_bi_losses[i]['bpr_loss'], label=f'Exp {i+1} - BI BPR Training Loss', linestyle='--', color='blue')
        plt.plot(epoch_list, all_bi_losses[i]['reg_loss'], label=f'Exp {i+1} - BI Reg Training Loss', linestyle='-.', color='blue')
        
        # KNN Losses
        plt.plot(epoch_list, all_knn_losses[i]['loss'], label=f'Exp {i+1} - KNN Total Training Loss', linestyle='-', color='orange')
        plt.plot(epoch_list, all_knn_losses[i]['bpr_loss'], label=f'Exp {i+1} - KNN BPR Training Loss', linestyle='--', color='orange')
        plt.plot(epoch_list, all_knn_losses[i]['reg_loss'], label=f'Exp {i+1} - KNN Reg Training Loss', linestyle='-.', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        #plt.legend()

        # Plot for metrics
        plt.subplot(1, 3, 2)
        # BI Metrics
        plt.plot(epoch_list, all_bi_metrics[i]['recall'], label=f'Exp {i+1} - BI Recall', linestyle='-', color='blue')
        plt.plot(epoch_list, all_bi_metrics[i]['precision'], label=f'Exp {i+1} - BI Precision', linestyle='--', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_knn_metrics[i]['recall'], label=f'Exp {i+1} - KNN Recall', linestyle='-', color='orange')
        plt.plot(epoch_list, all_knn_metrics[i]['precision'], label=f'Exp {i+1} - KNN Precision', linestyle='--', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('Recall & Precision')
        plt.title('Recall & Precision')
        
        # Plot for metrics
        plt.subplot(1, 3, 3)
        # BI Metrics
        plt.plot(epoch_list, all_bi_metrics[i]['ncdg'], label=f'Exp {i+1} - BI NCDG', linestyle='-', color='blue')
        
        # KNN Metrics
        plt.plot(epoch_list, all_knn_metrics[i]['ncdg'], label=f'Exp {i+1} - KNN NCDG', linestyle='-', color='orange')
        
        plt.xlabel('Epoch')
        plt.ylabel('NCDG')
        plt.title('NCDG')
        #plt.legend()

    # Custom Legend
    bi_line = mlines.Line2D([], [], color='blue', label='BI')
    knn_line = mlines.Line2D([], [], color='orange', label='KNN')
    plt.legend(handles=[bi_line, knn_line], loc='lower right')
    
    plt.tight_layout()  # Adjust spacing between subplots
    #plt.show()
    
    # Get current date and time
    now = datetime.now()

    # Format date and time as desired (e.g., "2024-08-27_14-30-00")
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(plot_name + '_' + timestamp +'.png')  # Save plot to file