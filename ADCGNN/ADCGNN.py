import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import sympy
import scipy
import numpy as np
from torch.nn import init

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.lin = lin 

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k] * feat
        return h

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x/2)**i * (1 - x/2)**(d - i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d+1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas

class ADCGNN_amazon(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False, dropout=0.2):
        super(ADCGNN_amazon, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.num_branches = len(self.thetas)
        self.conv = nn.ModuleList()
        for theta in self.thetas:
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, theta, lin=False))
            else:
                self.conv.append(PolyConvBatch(h_feats, h_feats, theta, lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attention_fc = nn.Linear(h_feats, 1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * h_feats, h_feats),
            nn.ReLU(),
            nn.Linear(h_feats, 1),
            nn.Sigmoid()
        )
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.residual_fc = nn.Linear(h_feats, h_feats)
        self.d = d

    def forward(self, in_feat):
        device = in_feat.device
        h_pre = self.linear(in_feat)
        h_pre = self.act(h_pre)
        h_pre = self.linear2(h_pre)
        h_pre = self.act(h_pre)
        h_pre = self.dropout(h_pre)
        res = self.residual_fc(h_pre)
        branch_outputs = []
        for conv in self.conv:
            h0 = conv(self.g, h_pre)
            h0 = h0.to(device)
            branch_outputs.append(h0)
        h_stack = torch.stack(branch_outputs, dim=1)
        N, B, F_dim = h_stack.shape
        h_flat = h_stack.view(-1, F_dim)              
        attn_scores = self.attention_fc(h_flat)        
        attn_scores = attn_scores.view(N, B)            
        attn_weights = F.softmax(attn_scores, dim=1)    
        
        attn_weights = attn_weights.unsqueeze(-1)       
        attn_fused = torch.sum(h_stack * attn_weights, dim=1) 
        mean_fused = torch.mean(h_stack, dim=1) 
        
        fusion_input = torch.cat([attn_fused, mean_fused], dim=-1)
        fusion_weight = self.fusion_mlp(fusion_input) 
        fused = 0.1*fusion_weight * attn_fused + (1 - fusion_weight) * mean_fused
        fused = fused + 0.8*res
        fused = self.linear3(fused)
        fused = self.act(fused)
        fused = self.dropout(fused)
        logits = self.linear4(fused)
        return logits


class ADCGNN_yelp(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super(ADCGNN_yelp, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        
        self.conv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.residual_fc = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        
        self.rel2id = {rel: idx for idx, rel in enumerate(self.g.canonical_etypes)}
        self.rel_emb = nn.Embedding(len(self.g.canonical_etypes), h_feats)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * h_feats, h_feats),
            nn.LeakyReLU(),
            nn.Linear(h_feats, 1, bias=False)
        )
        
        self.act = nn.LeakyReLU()

        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, in_feat):
        device = next(self.parameters()).device

        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        res = self.residual_fc(h)

        relation_hidden_list = []
        for relation in self.g.canonical_etypes:
            h_final = torch.zeros([len(in_feat), 0], device=device)
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h0 = h0.to(device)
                h_final = torch.cat([h_final, h0], dim=-1)
            h_relation = self.linear3(h_final)
            relation_hidden_list.append(h_relation)

        scores = []
        for i, h_relation in enumerate(relation_hidden_list):
            rel_idx = torch.tensor(i, dtype=torch.long, device=device)
            rel_vec = self.rel_emb(rel_idx) 
            rel_vec_expanded = rel_vec.unsqueeze(0).expand(h_relation.size(0), -1) 

            cat_feat = torch.cat([h_relation, rel_vec_expanded], dim=-1) 
            score_i = self.attn_mlp(cat_feat)
            scores.append(score_i)

        scores = torch.stack(scores, dim=0).squeeze(-1)
        attn_alpha = F.softmax(scores, dim=0)
        attn_alpha = attn_alpha.unsqueeze(-1)

        relation_hidden_stack = torch.stack(relation_hidden_list, dim=0)
        h_all = (relation_hidden_stack * attn_alpha).sum(dim=0)
        h_all = h_all + res
        h_all = self.act(h_all)
        out = self.linear4(h_all)
        return out

class ADCGNN_tfinance(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False, dropout=0.2):
        super(ADCGNN_tfinance, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.num_branches = len(self.thetas)
        self.conv = nn.ModuleList()
        for theta in self.thetas:
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, theta, lin=False))
            else:
                self.conv.append(PolyConvBatch(h_feats, h_feats, theta, lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attention_fc = nn.Linear(h_feats, 1)
        self.residual_fc = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.d = d

    def forward(self, in_feat):
        device = in_feat.device
        h_pre = self.linear(in_feat)
        h_pre = self.act(h_pre)
        h_pre = self.linear2(h_pre)
        h_pre = self.act(h_pre)
        h_pre = self.dropout(h_pre)
        res = self.residual_fc(h_pre)
        branch_outputs = []
        for conv in self.conv:
            h0 = conv(self.g, h_pre)
            branch_outputs.append(h0.to(device))
        h_stack = torch.stack(branch_outputs, dim=1)
        N, B, F_dim = h_stack.shape
        h_flat = h_stack.view(-1, F_dim)              
        attn_scores = self.attention_fc(h_flat)        
        attn_scores = attn_scores.view(N, B)          
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  
        attn_fused = torch.sum(h_stack * attn_weights, dim=1)      

        fused = attn_fused + 0.8*res
        fused = self.linear3(fused)
        fused = self.act(fused)
        fused = self.dropout(fused)
        logits = self.linear4(fused)
        return logits

class ADCGNN_CM(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super(ADCGNN_CM, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        
        self.conv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.residual_fc = nn.Linear(h_feats, h_feats)
        
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        
        self.rel_emb = nn.Embedding(len(self.g.canonical_etypes), h_feats)
        
        self.act = nn.LeakyReLU()

    def forward(self, in_feat):
        device = next(self.parameters()).device
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        res = self.residual_fc(h)
        relation_hidden_list = []
        for i, relation in enumerate(self.g.canonical_etypes):
            h_final = torch.zeros([len(in_feat), 0], device=device)
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0.to(device)], dim=-1)
            h_relation = self.linear3(h_final)
            relation_hidden_list.append(h_relation)
        scores = []
        for i, h_relation in enumerate(relation_hidden_list):
            rel_vec = self.rel_emb(torch.tensor(i, device=device))
            score_i = (h_relation * rel_vec.unsqueeze(0)).sum(dim=-1, keepdim=True)
            scores.append(score_i)
        scores = torch.stack(scores, dim=0).squeeze(-1)
        attn_alpha = F.softmax(scores, dim=0)  
        attn_alpha = attn_alpha.unsqueeze(-1)  
        relation_hidden_stack = torch.stack(relation_hidden_list, dim=0)
        h_all = torch.sum(relation_hidden_stack * attn_alpha, dim=0)
        h_all = h_all + res
        h_all = self.act(h_all)
        out = self.linear4(h_all)
        return out
