#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:24:35 2024

@author: salim
"""
import torch
from torch import nn
import torch.nn.functional as F




class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
    
class Residual(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, dim, dropout):
        super(Residual, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        "Residual connection with the same size."
        return x + self.dropout(self.norm(x))
    
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.mlp(x)
    
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = torch.Tensor([drop_prob])
        self.register_buffer('Drop out', self.drop_prob)

    def forward(self, x, training=None):
        if self.training:
            keep_prob = 1 - self.drop_prob
            shape = (x.size(0),) + (1,) * (x.dim() - 1)
            random_tensor = keep_prob + torch.rand(shape)
            random_tensor = random_tensor.floor()
            return (x / keep_prob) * random_tensor
        return x
    
class Embedding(nn.Module):
    def __init__(self, d_dim, embed_dim):
        super(Embedding, self).__init__()
        self.d_dim = d_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.d_dim, self.embed_dim)

    def forward(self, x):
        return self.embed(x)