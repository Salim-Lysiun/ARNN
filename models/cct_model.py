#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:16:05 2024

@author: salim
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange

#fLocal imports
from utils import Residual, MLP, StochasticDepth, Embedding

    
class SequencePooling(nn.Module):
    def __init__(self, in_features):
        super(SequencePooling, self).__init__()
        self.attention = nn.Linear(in_features, out_features=1)
        self.apply(self.init_weight)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        attention_weights = torch.transpose(attention_weights, 1, 2)
        weighted_representation = torch.matmul(attention_weights, x)
        return torch.squeeze(weighted_representation, dim=-2)

    @staticmethod
    def init_weight(m):
      if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.1)
        nn.init.constant_(m.bias,0)
        
        
class Conv_Tokenizer(nn.Module):
   def __init__(self):
    super(Conv_Tokenizer, self).__init__()
    self.conv = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 3, 3, 1, 1)),
        ('relu1', nn.ReLU()),
        ('max1', nn.MaxPool2d(3, 1, 1)),
        ('conv2', nn.Conv2d(3, 1, 3, 1, 1)),
        ('relu2', nn.ReLU()),
        ('max2', nn.MaxPool2d(3, 1, 1))]))
    self.apply(self.initialize_weight)

   def forward(self,x):
    return self.conv(x)

   @staticmethod
   def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight)
      
class PositionEmbedding(nn.Module):
    def __init__(self, sequence_length, dim):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.EmbeddingBag(sequence_length, dim)

    def forward(self, x):
        
        positions = torch.arange(x.size(-1)).expand(x.size(-2), -1)
        return x + self.embedding(positions)
    
    
class MHA(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()


        self.heads = heads
        self.dim = dim

        assert self.dim% self.heads==0, 'dim is not a factor of head_dim'
        self.head_dim = self.dim/self.heads

        inner_dim = int(self.head_dim *  self.heads)
        self.scale = self.head_dim  ** -0.5

        self.to_qkv = nn.Linear(self.dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, self.dim, bias= False),
            nn.Dropout(dropout)
        )
        self.apply(self.initialize_weight)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Attention Score
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

    @staticmethod
    def initialize_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
class CCT(nn.Module):
    def __init__(
        self,
        input_shape,
        embed_dim,
        num_heads,
        positional_emb=True,
        stochastic_depth_rate=0.1,
        dropout = 0.1,
        transformer_layers=6,
        num_classes=5,
    ):
        super(CCT, self).__init__()

        self.embedding = Embedding(input_shape[-1],embed_dim)  # Embedding of input tensor
        input_shape = (input_shape[0],input_shape[1], embed_dim)

        self.conv_tokenizer = Conv_Tokenizer()

        self.positional_emb = positional_emb
        if positional_emb:
            self.position_embedding = PositionEmbedding(sequence_length=input_shape[-2], dim = input_shape[-1])

        self.transformer_layers = transformer_layers
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout = dropout

        # Calculate Stochastic Depth probabilities.
        dpr = [x.item() for x in torch.linspace(0, self.stochastic_depth_rate, self.transformer_layers)]

        self.transformer_blocks = nn.Sequential()
        for i in range(transformer_layers):
            # Layer normalization 1.
            self.transformer_blocks.append(nn.LayerNorm(input_shape[-1]))

            # Create a multi-head attention layer.
            self.transformer_blocks.append(
                MHA(dim=input_shape[-1], heads=num_heads, dropout=0.1)
            )

            # Skip connection 1.
            self.transformer_blocks.append(StochasticDepth(dpr[i]))

            # Layer normalization 2.
            self.transformer_blocks.append(Residual(input_shape[-1],self.dropout))

            # MLP.
            self.transformer_blocks.append(MLP(dim=input_shape[-1], hidden_dim=20, dropout=self.dropout))

            # Skip connection 2.
            self.transformer_blocks.append(Residual(input_shape[-1],self.dropout))

        #self.ct_trans = nn.Sequential(*self.transformer_blocks)

        self.sequence_pooling = SequencePooling(input_shape[-1])  # Placeholder for SequencePooling

        self.classifier = nn.Linear(input_shape[-1], num_classes)

    def forward(self, x):
        # Embed data.
        data_embed = self.embedding(x)

        # Convolution Tokenizer.
        data_conv = self.conv_tokenizer(data_embed.unsqueeze(1)).squeeze()

        # Apply positional embedding.
        if self.positional_emb:
            data_conv = self.position_embedding(data_conv)

        data = self.transformer_blocks(data_conv)

        # Apply sequence pooling.
        weighted_representation = self.sequence_pooling(data)

        # Classify outputs.
        out = self.classifier(weighted_representation)

        return out


