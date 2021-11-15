import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import random
import os
from torch.autograd import Variable
import dgl.function as fn
from sklearn.metrics import roc_auc_score,precision_score, recall_score, f1_score,accuracy_score,average_precision_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(101) 

class fusion(nn.Module):
    def __init__(self, infeats,hid_feat,out_feat, reduction=16):
        super(fusion1, self).__init__()

        self.fc1=nn.Linear(infeats*2, infeats*2 // reduction, bias=False)
        self.fc2= nn.Linear(infeats*2 // reduction, infeats, bias=False)
        self.fc3= nn.Linear(infeats*2 // reduction, infeats, bias=False)

        self.mlp=nn.Sequential(
            nn.Linear(in_feats*2, hid_feat),
            nn.ReLU(inplace=True),
            nn.Linear(hid_feat, hid_feat),
            nn.ReLU(inplace=True),
            nn.Linear(hid_feat, out_feat)            
        )
        self.sig=nn.Sigmoid()
    def Norm(self,inp):
        return (inp-inp.mean(axis=1).view(-1,1))/inp.std(axis=1).view(-1,1)  
    
    def forward(self, x,y,inputs):
        
        xy1=self.fc1(inputs)
        xy1=F.relu(xy1)
        xy2=self.fc2(xy1)
        xy3=self.fc3(xy1)
        xy2=self.sig(xy2)
        xy3=self.sig(xy3)
        xl=self.Norm(x*xy2)
        yl=self.Norm(y*xy3)

        xyl=torch.cat([xl,yl],1)

        xyl=self.mlp(xyl)
        
        return torch.sigmoid(xyl)



def binary_loss(pos_logits, neg_logits):

    pre = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones(pos_logits.shape[0]), torch.zeros(neg_logits.shape[0])])
    # labels=labels

    loss_func=torch.nn.BCELoss()
    loss=loss_func(pre,labels)

    return loss


def binary_val(model,input_x,input_y,inputs,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        logits = model(input_x,input_y,inputs)

        pos_val = logits[val_pos_mask].view(-1)
        neg_val = logits[val_neg_mask].view(-1)

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])])
        # loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        loss=binary_loss(pos_val,neg_val)
        # loss=F.cross_entropy(pre, labels)
        auc=roc_auc_score(labels, pre)
        # _, indices = torch.max(pre, dim=1)
        # correct = torch.sum(indices == labels)
        return loss,auc        

def compute_auc(pos_score, neg_score):

    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()

    auc=roc_auc_score(labels, scores)
    # precision=precision_score(labels, scores)
    # recall=recall_score(labels, scores)
    # f1=f1_score(labels, scores)
    # accuracy=accuracy_score(labels, scores)

    return auc
