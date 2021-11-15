import networkx as nx
import matplotlib.pyplot as plt
import dgl
import numpy as np
import pandas as pd
import urllib.request
from dgl.data import DGLDataset
import torch
import os
import random
import scipy.stats as stats
from scipy.stats.stats import pearsonr
import itertools
import umap
from sklearn.metrics import pairwise_distances  
from sklearn.cross_decomposition import CCA,PLSSVD
def load_data(name):
    express_path='./data/expression/'+name+'_expression.tsv'
    network_path='./data/network/'+name+'_network.tsv'
#     ranking_path='./data/ranking/'+name+'_ranking.txt'
    tfs_path='./data/tfs/'+name+'_transcription_factors.tsv'
    
#     if name=='silico':
#         rg_type='./data/regulation_type/'+name+'_regualtion_type.tsv'
    
    expression=pd.read_csv(express_path, sep='\t')
    network=pd.read_csv(network_path, sep='\t', header=None)
#     ranking=pd.read_csv(ranking_path, sep='\t', header=None)
    tfs=pd.read_csv(tfs_path, sep='\t', header=None)
    
    return expression,network,tfs

def load_data_new(name):
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
def int_edges(net,nodes_arr):
    edges=[]
    for i in net:
        u=np.where(nodes_arr==i[0])[0][0]
        v=np.where(nodes_arr==i[1])[0][0]
        edges.append([i[0],i[1],u,v,i[2]])
    return edges
def adj_list(i_feat,min_k=2,metric='euclidean',dr=False):
    # output
    if dr:
        data=umap.UMAP(n_components=100, min_dist=0.00001).fit_transform(i_feat)
    else:
        data=i_feat

    if metric == 'pearson' :
        
        df = pd.DataFrame(data.T)
        corr = df.corr(method='pearson')
        pairs_r = 1-abs(corr)
        # pairs_r = 1-df.corr(method='pearson')
        # pairs_r = 1-abs(df.corr(method='pearson'))
        adj_m = pairs_r.values
    else:
        adj_m = pairwise_distances(data, metric=metric)
        
    adj_list=[]
    adj_row=[]
    for i,j in enumerate(adj_m):
        min_k_idx=j.argsort()[1:min_k+1]
        adj_row.append([i]+list(min_k_idx))    
        for k in min_k_idx:
            
            adj_list.append([k,i])
    # adj_list=[]
    # for i,j in enumerate(adj_m):
    # #     min_k_idx=j.argsort()[1:min_k+1]
    #     negtive_k_cor=j.argsort()[0:min_k]
    #     positive_k_cor=j.argsort()[(-1-min_k):-1]
        
    #     min_k_idx=np.hstack((negtive_k_cor,positive_k_cor))
    #     for k in min_k_idx:
            
    #         adj_list.append([i,k])   
    return adj_list,adj_m,abs(corr)

def double_v_cca(v1,v2,fusion_type='concat'):
    v1=v1.reshape(len(v1),1)
    v2=v2.reshape(len(v1),1)
    cca = CCA(n_components=1)
    cca.fit(v1, v2)
    
    v1_c, v2_c = cca.transform(v1, v2)
    corr=np.corrcoef(v1_c[:, 0], v2_c[:, 0])[0,1]
    
    v1_c=v1_c.flatten()
    v2_c=v2_c.flatten()
    if fusion_type=='concat':
        z=np.hstack((v1_c,v2_c))
    if fusion_type=='sum':
        z=v1_c+v2_c
#     z=pca_trans(v1_c,v2_c)
    return z,corr

def create_fake_data(ranking,data,q=0.5):
    
    t=np.quantile(np.unique(ranking[2].values),q,interpolation='higher')
    fake_edges=ranking[ranking[2]>=t]
    f_nodes=np.unique(fake_edges[[0,1]].values.flatten())
    f_ex=data[f_nodes]
    
    f_exp=[]
    for i in f_ex.values:
        rv = stats.beta(a=0.5, b=0.5)
        f_exp.append(list(rv.cdf(i)))
    f_expression=pd.DataFrame(np.array(f_exp),columns=f_ex.columns)
    
    f_edges=[]
    for i in fake_edges.values:
        s=i[0].replace('G','F')
        d=i[1].replace('G','F')
        r=i[2]
        f_edges.append([s,d,r])
    f_edges=pd.DataFrame(f_edges)
    
    col=[]
    for i in f_expression.columns:
        col.append(i.replace('G','F'))

    f_expression2=pd.DataFrame(f_expression.values,columns=col)
    f_ex=pd.DataFrame(f_ex.values,columns=col)


    return f_edges,f_expression2,f_ex

def edges_and_expression(edges,expression):
    # all_nodes=np.unique(edges[[0,1]].values.flatten())
    all_nodes=expression.columns
    nodes_index=pd.DataFrame(all_nodes).index.values
    nodes_list=np.vstack((all_nodes,nodes_index)).T
    all_edges=edges[[0,1]]
    str_2_num=all_edges.copy()
    for i in nodes_list:
        str_2_num.loc[str_2_num[0] ==i[0], 0] = i[1]
        str_2_num.loc[str_2_num[1] ==i[0], 1] = i[1]
    edges_index_list=pd.concat([all_edges, str_2_num], axis=1).values
    expression_data=expression[nodes_list[:,0]]
    expression_data.columns=nodes_list[:,1]
    return {'node':nodes_list,'edges':edges_index_list,'expression':expression_data}

def construct_neg_edges(pos_edge):
    
    nodes_name=np.unique(pos_edge)
    adj=pd.DataFrame(0,columns=nodes_name,index=nodes_name)
    for i in pos_edge:
        adj[i[0]][i[1]]=1
    neg_u, neg_v = np.where(adj==0)
    neg_u=adj.columns[neg_u]
    neg_v=adj.columns[neg_v]
    eids=np.arange(len(neg_u))
    eids = np.random.permutation(eids)
    eids= eids[:len(pos_edge)]
    neg_u,neg_v=neg_u[eids],neg_v[eids]
    
    return neg_u,neg_v

def construct_neg_edges2(pos_edge,num_of_all_nodes):
    
    nodes_name=np.unique(pos_edge)
    n=np.arange(0,num_of_all_nodes)
    adj=pd.DataFrame(0,columns=n,index=n)
    for i in pos_edge:
        adj[i[0]][i[1]]=1
    neg_u, neg_v = np.where(adj==0)
    neg_u=adj.columns[neg_u]
    neg_v=adj.columns[neg_v]
    # eids=np.arange(len(neg_u))
    # eids = np.random.permutation(eids)
    # eids= eids[:len(pos_edge)]
    # neg_u,neg_v=neg_u[eids],neg_v[eids]
    
    return neg_u,neg_v    

def pos_data_split(num_of_all_nodes,pos_edges,test_ratio=0.2):
# def train_test_split(num_of_all_nodes,pos_edges,neg_edges,test_ratio=0.2):    
    u,v=pos_edges[:,0],pos_edges[:,1]
    
    eids=np.arange(len(u))
    eids = np.random.permutation(eids)
    val_size=int(len(eids) * 0.1)
    
    test_size = int(len(eids) * test_ratio)
    train_size = len(u) - (test_size+val_size)
    
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u,val_pos_v=u[eids[test_size:test_size+val_size]], v[eids[test_size:test_size+val_size]]
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]
    
    # if len(neg_edges)<0:
    #     neg_u, neg_v = construct_neg_edges2(pos_edges)
    # if len(neg_edges)>0:
    #     neg_u, neg_v =neg_edges[:,0],neg_edges[:,1]
    # neg_u, neg_v = construct_neg_edges2(pos_edges,num_of_all_nodes)

    # # neg_u, neg_v =neg_edges[:,0],neg_edges[:,1]
    # neg_eids = np.random.choice(len(neg_u), len(u))
    
    # test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    # val_neg_u,val_neg_v=neg_u[neg_eids[test_size:test_size+val_size]], neg_u[neg_eids[test_size:test_size+val_size]]
    # train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]], neg_v[neg_eids[test_size+val_size:]]
    
    train_pos_g = dgl.graph((train_pos_u.astype(int), train_pos_v.astype(int)), num_nodes=num_of_all_nodes)
    # train_neg_g = dgl.graph((train_neg_u.astype(int), train_neg_v.astype(int)), num_nodes=num_of_all_nodes)

    val_pos_g = dgl.graph((val_pos_u.astype(int), val_pos_v.astype(int)), num_nodes=num_of_all_nodes)
    # val_neg_g = dgl.graph((val_neg_u.astype(int), val_neg_v.astype(int)), num_nodes=num_of_all_nodes)    
    
    
    test_pos_g = dgl.graph((test_pos_u.astype(int), test_pos_v.astype(int)), num_nodes=num_of_all_nodes)
    # test_neg_g = dgl.graph((test_neg_u.astype(int), test_neg_v.astype(int)), num_nodes=num_of_all_nodes)
    
    graph_s={'train':train_pos_g,'val':val_pos_g,'test':test_pos_g}
    # return train_pos_g,train_neg_g,val_pos_g,val_neg_g,test_pos_g,test_neg_g
    return graph_s

def train_test_split(num_of_all_nodes,pos_edges,test_ratio=0.2):
# def train_test_split(num_of_all_nodes,pos_edges,neg_edges,test_ratio=0.2):    
    u,v=pos_edges[:,0],pos_edges[:,1]
    
    eids=np.arange(len(u))
    eids = np.random.permutation(eids)
    val_size=int(len(eids) * 0.1)
    
    test_size = int(len(eids) * test_ratio)
    train_size = len(u) - (test_size+val_size)
    
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u,val_pos_v=u[eids[test_size:test_size+val_size]], v[eids[test_size:test_size+val_size]]
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]], v[eids[test_size+val_size:]]
    
    # if len(neg_edges)<0:
    #     neg_u, neg_v = construct_neg_edges2(pos_edges)
    # if len(neg_edges)>0:
    #     neg_u, neg_v =neg_edges[:,0],neg_edges[:,1]
    neg_u, neg_v = construct_neg_edges2(pos_edges,num_of_all_nodes)

    # neg_u, neg_v =neg_edges[:,0],neg_edges[:,1]
    neg_eids = np.random.choice(len(neg_u), len(u))
    
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u,val_neg_v=neg_u[neg_eids[test_size:test_size+val_size]], neg_u[neg_eids[test_size:test_size+val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]], neg_v[neg_eids[test_size+val_size:]]
    
    train_pos_g = dgl.graph((train_pos_u.astype(int), train_pos_v.astype(int)), num_nodes=num_of_all_nodes)
    train_neg_g = dgl.graph((train_neg_u.astype(int), train_neg_v.astype(int)), num_nodes=num_of_all_nodes)

    val_pos_g = dgl.graph((val_pos_u.astype(int), val_pos_v.astype(int)), num_nodes=num_of_all_nodes)
    val_neg_g = dgl.graph((val_neg_u.astype(int), val_neg_v.astype(int)), num_nodes=num_of_all_nodes)    
    
    
    test_pos_g = dgl.graph((test_pos_u.astype(int), test_pos_v.astype(int)), num_nodes=num_of_all_nodes)
    test_neg_g = dgl.graph((test_neg_u.astype(int), test_neg_v.astype(int)), num_nodes=num_of_all_nodes)
    
    graph_s={'train':[train_pos_g,train_neg_g],'val':[val_pos_g,val_neg_g],'test':[test_pos_g,test_neg_g]}
    # return train_pos_g,train_neg_g,val_pos_g,val_neg_g,test_pos_g,test_neg_g
    return graph_s
def neg_g_sampling(graph,neg_edges,k,rdm=False):
    
    src, dst = graph.edges()
    if rdm:
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))  
        # neg_edges=np.random.permutation(neg_edges)
        # neg_sample=neg_edges[:len(src)]
        # neg_src,neg_dst=neg_sample[:,0],neg_sample[:,1]
        
    else:
        src=src.numpy()
        dst=dst.numpy()
        neg_elist=[]
        for i in src:
            ngs=neg_edges[neg_edges[:,0]==i]
            negs=np.random.permutation(ngs)
            neg_edge=negs[:k]
            # neg_edge=negs
            for j in neg_edge:
                neg_elist.append(list(j))
        neg_src,neg_dst=np.array(neg_elist)[:,0],np.array(neg_elist)[:,1]
        
    neg_g=dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())
    return neg_g
def subs_sampling(full_list,shuffle=False,n=5):
    n_total = len(full_list)
    offset = round(n_total/n)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        np.random.shuffle(full_list)         
        sub_list= [full_list[i:i+offset] for i in range(0,len(full_list),offset)]
        if len(sub_list)==n:
            return sub_list
        elif len(sub_list)==n+1:
            print(len(sub_list))
            g_index=random.choice(range(n-1))
            sub_list[g_index]=np.r_[sub_list[g_index],sub_list[n]]
            del sub_list[n]
            return sub_list
        else:
            print('Warning!!!')
            print('Error: sub_list leng th =',len(sub_list),',unspport split amount')

def sign_graph_id(edges,sub_g_list):
    g_id=np.zeros(len(edges[:,0]))
    for i,j in enumerate ( sub_g_list):
        for k in j:
            index=np.where(edges[:,0]==k)[0]
            for m in index:
                g_id[m]=i
    g_id=np.reshape(g_id, (-1, 1))
    edges_with_id=np.concatenate((edges,g_id),axis=1)
    
    return edges_with_id            


def sign_graph_id_edges(edges,sub_g_list):

    g_id=np.zeros(len(edges[:,0]))
    for i,j in enumerate (sub_g_list):
        index=np.where((edges==j[:,None]).all(-1))[1]
    #     print(index)
        g_id[index]=i    
    g_id=np.reshape(g_id, (-1, 1))
    edges_with_id=np.concatenate((edges,g_id),axis=1)
    
    return edges_with_id  


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
    # test_neg_id=np.setdiff1d(neg_id,test_pos_id)
    # test_neg_id=np.setdiff1d(test_neg_id,train_neg_id)
    # test_neg_id=np.setdiff1d(neg_id,val_neg_id)
    # test_neg_id = np.random.permutation(test_neg_id)
    test_neg_id=neg_id[val_size:test_size+val_size]    

    # train_neg_id_set=np.setdiff1d(neg_id,val_neg_id)
    # train_neg_id_set=np.setdiff1d(train_neg_id_set,test_neg_id)
    train_neg_id_set=neg_id[test_size+val_size:]   
    # train_neg_mask[train_neg_id]=True
    test_neg_mask[test_neg_id]=True
    val_neg_mask[val_neg_id]=True    

    return test_neg_mask,val_neg_mask, train_neg_id_set

def neg_masks(pos_number,negtive_number, test_pos_id,test_ratio):
    mask=np.full((pos_number+negtive_number), False)
    
    neg_id=np.arange(pos_number,negtive_number)
    neg_id=np.hstack((neg_id,test_pos_id))
    neg_id=np.random.permutation(neg_id)
    
    
    val_size=int(pos_number * 0.1)
    test_size = int(pos_number * test_ratio)
    train_size = pos_number - (test_size+val_size)
    
    train_neg_mask=mask.copy()
    test_neg_mask=mask.copy()
    val_neg_mask=mask.copy()
    

    train_neg_id=neg_id[:train_size]
    val_neg_id=neg_id[train_size:train_size+val_size]
    
    test_neg_id=np.setdiff1d(neg_id,test_pos_id)
    test_neg_id=np.setdiff1d(test_neg_id,train_neg_id)
    test_neg_id=np.setdiff1d(test_neg_id,val_neg_id)
    test_neg_id = np.random.permutation(test_neg_id)
    test_neg_id=test_neg_id[:test_size]
    
    train_neg_mask[train_neg_id]=True
    test_neg_mask[test_neg_id]=True
    val_neg_mask[val_neg_id]=True
    
    return {'neg_masks':[train_neg_mask,test_neg_mask,val_neg_mask]}    


def train_neg_mask_build(pos_number,negtive_number,train_neg_set,train_size):
#     train_size = pos_number - (test_ratio+val_ratio)
    train_neg_id=np.random.permutation(train_neg_set)
    train_neg_id=train_neg_id[:train_size*20]
    
    train_neg_mask=np.full((pos_number+negtive_number), False)
    train_neg_mask[train_neg_id]=True    
    
    return train_neg_mask    

# def scorer2_train_neg_mask_builder(pos_number,negtive_number,train_neg_size,train_neg_ids):
#     neg_ids=np.random.permutation(train_neg_ids)
#     train_neg_id=neg_ids[:train_neg_size]

#     train_neg_mask=np.full((pos_number+negtive_number), False)
#     train_neg_mask[train_neg_id]=True
#     return train_neg_mask
def known_split(pos_number,negtive_number,known_ratio):

    mask=np.full((pos_number+negtive_number), False)
    pos_id=np.arange(pos_number)
    pos_id=np.random.permutation(pos_id)

    known_pos_size=int(pos_number*0.9)
    test_pos_size=pos_number-known_pos_size

    known_pos_mask=mask.copy()
    test_pos_mask=mask.copy()


    known_pos_id=pos_id[:known_pos_size]
    test_pos_id=pos_id[known_pos_size:]

    known_pos_mask[known_pos_id]=True
    test_pos_mask[test_pos_id]=True

    unknown_as_neg_mask=np.full((pos_number+negtive_number), True)
    unknown_as_neg_mask[known_pos_id]=False
    unknown_as_neg_id=np.where(unknown_as_neg_mask==True)[0]

    
    return {'pos_ids':[known_pos_id,test_pos_id,unknown_as_neg_id],
            'masks':[known_pos_mask,test_pos_mask,unknown_as_neg_mask]}