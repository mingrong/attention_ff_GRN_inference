import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import random
import os
from torch.autograd import Variable
# from dgl.nn import SAGEConv
from sage import SAGEConv
from dgl.nn.pytorch import GATConv
import dgl.function as fn
from sklearn.metrics import roc_auc_score,precision_score, recall_score, f1_score,accuracy_score,average_precision_score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(108) 

class dnn(nn.Module):
    def __init__(self,
                infeat,
                hid_feat,
                out_feat):
        super(dnn, self).__init__()
        self.layer1=nn.Linear(infeat, hid_feat)
        self.layer2=nn.Linear(hid_feat, hid_feat)
        self.layer3=nn.Linear(hid_feat, out_feat)
    
    def Norm(self,inp):
        return (inp-inp.mean(axis=1).view(-1,1))/inp.std(axis=1).view(-1,1)  

    def forward(self, x):
        x = self.layer1(x)
        # x = self.Norm(x)
        x=F.relu(x)
        x = self.layer2(x)
        # x = self.Norm(x)
        x=F.relu(x)
        x = self.layer3(x)
        return torch.sigmoid(x)

class dot_dnn(nn.Module):
    def __init__(self,
                infeat,
                hid_feat,
                out_feat):
        super(dot_dnn, self).__init__()
        self.src1=nn.Linear(infeat,hid_feat)
        self.src2=nn.Linear(hid_feat,hid_feat)
        self.src3=nn.Linear(hid_feat,out_feat)
        
        self.dst1=nn.Linear(infeat,hid_feat)
        self.dst2=nn.Linear(hid_feat,hid_feat)
        self.dst3=nn.Linear(hid_feat,out_feat)        
        

    def forward(self,src_data,dst_data):
        
        x=self.src1(src_data)
        x=F.relu(x)
        x=self.src2(x)
        x=F.relu(x)
        x=self.src3(x)
        
        y=self.dst1(dst_data)
        y=F.relu(y)
        y=self.dst2(y)
        y=F.relu(y)
        y=self.dst3(y)
        
        score=torch.sum(x*y,1)
        
        return score

class VAE(nn.Module):
    def __init__(self,
                infeat,
                hid_feat,
                hid_feat2,
                out_dim
                ):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(infeat, hid_feat)
        self.fc21 = nn.Linear(hid_feat, hid_feat2)
        self.fc22 = nn.Linear(hid_feat, hid_feat2)
        self.fc3 = nn.Linear(hid_feat2, hid_feat)
        self.fc4 = nn.Linear(hid_feat, infeat)
        self.classifier=dnn(infeat,128,2)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # if torch.cuda.is_available():
        #     eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # print(z.shape)
        de_z=self.decode(z)
        pre=self.classifier(de_z)
        return de_z, mu, logvar,pre

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'gcn')
        self.conv2 = SAGEConv(h_feats, h_feats, 'gcn')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self,in_dim, hidden_dim, out_dim, num_heads,feat_drop=0.,attn_drop=0.,allow_zero_in_degree=True):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads,feat_drop=feat_drop,attn_drop=attn_drop,allow_zero_in_degree=True)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1,allow_zero_in_degree=True)


    def forward(self, g,h):
        h = self.layer1(g,h)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.relu(h)
        h = self.layer2(g,h)
        h = h.squeeze()
        return h

class MLPPredictor(nn.Module):
    def __init__(self, in_feature, hid_feature,out_classes):
        super().__init__()
        self.layer1=nn.Linear(in_feature*2, hid_feature)
        self.layer2=nn.Linear(hid_feature, hid_feature)
        self.layer3=nn.Linear(hid_feature, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        
        h = torch.cat([h_u, h_v], 1)
        x = self.layer1(h)
        x=F.relu(x)
        x = self.layer2(x)
        x=F.relu(x)
        x = self.layer3(x)
        # score=torch.sum(h_u*h_v, 1)

        return {'score': x}
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class gnn(nn.Module):
    def __init__(self,
                in_feats,
                n_hidden,
                out_dim,
                n_layers,
                norm,
                activation,
                dropout,
                aggregator_type):
        super(gnn, self).__init__()

        self.layers=nn.ModuleList()

        self.layers.append(SAGEConv(in_feats=in_feats, 
                                    out_feats=n_hidden, 
                                    aggregator_type=aggregator_type,
                                    feat_drop=dropout, 
                                    activation=activation,
                                    norm=norm))
        for i in range(n_layers-1):
            self.layers.append(SAGEConv(in_feats=n_hidden, 
                                    out_feats=n_hidden, 
                                    aggregator_type=aggregator_type,
                                    feat_drop=dropout, 
                                    activation=activation,
                                    norm=norm))  

        self.layers.append(SAGEConv(in_feats=n_hidden, 
                                    out_feats=out_dim, 
                                    aggregator_type=aggregator_type,
                                    # feat_drop=dropout, 
                                    activation=None))
        self.pre=MLPPredictor(out_dim,128,2)
        # self.pre=DotPredictor()

    def forward(self, graph, pos_g,neg_g,inputs):
        h=inputs
        for layer in self.layers:
            h=layer(graph,h)
            # h=F.relu(h)
        # h=self.layer(graph,h)
        pos_result=self.pre(pos_g,h) 
        neg_result=self.pre(neg_g,h)   
        return pos_result,neg_result

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


# def compute_loss(pos_score, neg_score):

#     scores = torch.cat([pos_score, neg_score])
#     labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
#     # return F.cross_entropy(scores, labels)
#     return F.binary_cross_entropy_with_logits(scores, labels)
def batch_loss(logits,labels):
    loss_func=torch.nn.BCELoss()
    loss=loss_func(logits,labels)

    return loss
def batch_val(model,inputs,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():

        val_set=torch.cat([inputs[val_pos_mask],inputs[val_neg_mask]],0)

        logits = model(val_set)
        logits=logits.view(-1)
        n=int(logits.shape[0]/2) 

        pos_val = logits[:n]
        neg_val = logits[n:]

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])])
        # loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        loss=batch_loss(pre,labels)
        # loss=F.cross_entropy(pre, labels)
        auc=roc_auc_score(labels, pre)
        # _, indices = torch.max(pre, dim=1)
        # correct = torch.sum(indices == labels)
        return loss,auc
def self_att_loss(logits):
    logits=logits.view(-1)
    n=int(logits.shape[0]/2)
    pos_logits=logits[:n]
    neg_logits=logits[n:]

    pre = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones(pos_logits.shape[0]), torch.zeros(neg_logits.shape[0])])
    # labels=labels

    loss_func=torch.nn.BCELoss()
    loss=loss_func(pre,labels)

    return loss
def self_att_val(model,inputs,val_pos_mask,val_neg_mask):    
    model.eval()
    with torch.no_grad():

        val_set=torch.cat([inputs[val_pos_mask],inputs[val_neg_mask]],0)

        logits = model(val_set)
        logits=logits.view(-1)
        n=int(logits.shape[0]/2)

        pos_val = logits[:n]
        neg_val = logits[n:]

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])])
        # loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        loss=self_att_loss(logits)
        # loss=F.cross_entropy(pre, labels)
        auc=roc_auc_score(labels, pre)
        # _, indices = torch.max(pre, dim=1)
        # correct = torch.sum(indices == labels)
        return loss,auc
def binary_loss(pos_logits, neg_logits):

    pre = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones(pos_logits.shape[0]), torch.zeros(neg_logits.shape[0])])
    # labels=labels

    loss_func=torch.nn.BCELoss()
    loss=loss_func(pre,labels)

    return loss

def binary_val(model,inputs,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)

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
def binary_val2(model,inputs,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)

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
def binary_val3(model,input_x,input_y,inputs,val_pos_mask,val_neg_mask):
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
def compute_loss(pos_logits, neg_logits):

    pre = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones(pos_logits.shape[0]), torch.zeros(neg_logits.shape[0])]).type(torch.LongTensor)
    # labels=labels
    _, indices = torch.max(pre, dim=1)
    correct = torch.sum(indices == labels)

    return F.cross_entropy(pre, labels),correct.item() * 1.0 / len(labels)

# def     
def compute_score_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
#     print(type(labels))
    # return F.cross_entropy(scores, labels)
    return F.binary_cross_entropy_with_logits(scores,labels)
def compute_interval_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
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
def vae_loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = F.mse_loss(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

def evaluate_vae(model, features,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        recon_x,mu,logvar, logits = model(features)
        loss1 = vae_loss_function(recon_x, features, mu, logvar)
        pos_val = logits[val_pos_mask]
        neg_val = logits[val_neg_mask]

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])]).type(torch.LongTensor)

        loss2=F.cross_entropy(pre, labels)

        _, indices = torch.max(pre, dim=1)
        correct = torch.sum(indices == labels)

        return loss2,correct.item() * 1.0 / len(labels)    
def evaluate(model,graph,val_pos_g,val_neg_g,features):
    model.eval()
    with torch.no_grad():
        # logits = model(features)

        # pos_val = features[val_pos_mask]
        # neg_val = features[val_neg_mask]

        # pre=torch.cat([pos_val, neg_val])
        # labels = torch.cat(
        #     [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])]).type(torch.LongTensor)
        val_pos_logits,val_neg_logits=model(graph,val_pos_g,val_neg_g,features)
        pre=torch.cat([val_pos_logits, val_neg_logits])
        labels = torch.cat(
            [torch.ones(val_pos_logits.shape[0]), torch.zeros(val_neg_logits.shape[0])]).type(torch.LongTensor)

        loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        # loss=F.cross_entropy(pre, labels)
        auc=roc_auc_score(labels, pre)
        # _, indices = torch.max(pre, dim=1)
        # correct = torch.sum(indices == labels)
        return loss,auc
def val_gnn_lc(model,graph,val_pos_g,val_neg_g,features):

    model.eval()
    with torch.no_grad():

        val_pos_logits,val_neg_logits=model(graph,val_pos_g,val_neg_g,features)
        pre=torch.cat([val_pos_logits, val_neg_logits])
        labels = torch.cat(
            [torch.ones(val_pos_logits.shape[0]), torch.zeros(val_neg_logits.shape[0])]).type(torch.LongTensor)
        loss=F.cross_entropy(pre, labels)

        _, indices = torch.max(pre, dim=1)
        correct = torch.sum(indices == labels)
        return loss,correct.item() * 1.0 / len(labels)   

def evaluate_binary_c(model,features,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)

        pos_val = logits[val_pos_mask]
        neg_val = logits[val_neg_mask]

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])]).type(torch.LongTensor)
        # loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        loss=F.cross_entropy(pre, labels)
        # auc=roc_auc_score(labels, pre)
        _, indices = torch.max(pre, dim=1)
        correct = torch.sum(indices == labels)
        return loss,correct.item() * 1.0 / len(labels)   
def evaluate_dot(model,x,y,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        logits = model(x,y)

        pos_val = logits[val_pos_mask]
        neg_val = logits[val_neg_mask]

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])]).type(torch.LongTensor)
        # loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        loss=compute_interval_loss(pos_val,neg_val)
        # loss=F.cross_entropy(pre, labels)
        auc=roc_auc_score(labels, pre)
        # _, indices = torch.max(pre, dim=1)
        # correct = torch.sum(indices == labels)
        return loss,auc
        # return loss,correct.item() * 1.0 / len(labels)  
def evaluate_linear(model,x,y,val_pos_mask,val_neg_mask):
    model.eval()
    with torch.no_grad():
        logits = model(x,y)

        pos_val = logits[val_pos_mask]
        neg_val = logits[val_neg_mask]

        pre=torch.cat([pos_val, neg_val])
        labels = torch.cat(
            [torch.ones(pos_val.shape[0]), torch.zeros(neg_val.shape[0])]).type(torch.LongTensor)
        # loss=F.binary_cross_entropy_with_logits(pre,labels.float())
        loss=F.cross_entropy(pre, labels)
        # auc=roc_auc_score(labels, pre)
        _, indices = torch.max(pre, dim=1)
        correct = torch.sum(indices == labels)
        return loss,correct.item() * 1.0 / len(labels)
def validation(val_pos_g,val_neg_g,h,pred):
    # pred = DotPredictor()

    with torch.no_grad():

        pos_score = pred(val_pos_g, h).flatten()
        neg_score = pred(val_neg_g, h).flatten()
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()

        auc=roc_auc_score(labels, scores)
        aupr=average_precision_score(labels, scores)
        return compute_score_loss(pos_score, neg_score),auc,aupr
        # return compute_loss(pos_score, neg_score),auc,aupr



