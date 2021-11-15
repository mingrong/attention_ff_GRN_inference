import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request
import torch
import os
import random
import scipy.stats as stats
from scipy.stats.stats import pearsonr
import itertools


def load_data(name):
    express_path='./data/expression/'+name+'_expression.tsv'
    network_path='./data/network/net_with_int/'+name+'_network.csv'
#     ranking_path='./data/ranking/'+name+'_ranking.txt'
    tfs_path='./data/tfs/'+name+'_transcription_factors.tsv'
    
#     if name=='silico':
#         rg_type='./data/regulation_type/'+name+'_regualtion_type.tsv'
    
    expression=pd.read_csv(express_path, sep='\t')
    network=pd.read_csv(network_path, header=None)
#     ranking=pd.read_csv(ranking_path, sep='\t', header=None)
    tfs=pd.read_csv(tfs_path, sep='\t', header=None)
    
    return expression,network,tfs    


def pos_masks(pos_number,negtive_number, test_ratio):
    mask=np.full((pos_number+negtive_number), False)
    
    pos_id=np.arange(pos_number)
    pos_id=np.random.permutation(pos_id)

    
    val_size=int(len(pos_id) * 0.1)
    test_size = int(len(pos_id) * test_ratio)
    train_size = len(pos_id) - (test_size+val_size)
    
    train_pos_mask=mask.copy()
    test_pos_mask=mask.copy()
    val_pos_mask=mask.copy()
    
    train_pos_mask[pos_id[:train_size]]=True
    test_pos_mask[pos_id[train_size:train_size+test_size]]=True
    val_pos_mask[pos_id[train_size+test_size:]]=True
    
    test_id=pos_id[train_size:train_size+test_size]
    return {'pos_masks':[train_pos_mask,test_pos_mask,val_pos_mask]
           ,'test_id':test_id,'train_size':train_size}

def test_val_neg_maks(pos_number,negtive_number,test_ratio=0.1,val_ratio=0.1):

    mask=np.full((pos_number+negtive_number), False)
    neg_id=np.arange(pos_number,negtive_number)
    # neg_id=np.hstack((neg_id,test_pos_id))
    neg_id=np.random.permutation(neg_id)


    val_size=int(pos_number * val_ratio)
    test_size = int(pos_number * test_ratio)
    train_size = pos_number - (test_size+val_size)

    test_neg_mask=mask.copy()
    val_neg_mask=mask.copy()    


    val_neg_id=neg_id[:val_size]
    test_neg_id=neg_id[val_size:test_size+val_size]    


    train_neg_id_set=neg_id[test_size+val_size:]   
    test_neg_mask[test_neg_id]=True
    val_neg_mask[val_neg_id]=True    

    return test_neg_mask,val_neg_mask, train_neg_id_set
def train_neg_mask_build(pos_number,negtive_number,train_neg_set,train_size):
#     train_size = pos_number - (test_ratio+val_ratio)
    train_neg_id=np.random.permutation(train_neg_set)
    train_neg_id=train_neg_id[:train_size]
    
    train_neg_mask=np.full((pos_number+negtive_number), False)
    train_neg_mask[train_neg_id]=True    
    
    return train_neg_mask

