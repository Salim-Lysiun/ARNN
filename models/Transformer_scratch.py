#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:26:35 2023

@author: salim
"""
"""
Paper Title: " TLSTM: A hybrid model for multi-channel EEG signals classification for seizures"

Model : Transformer Model from scratch Implementation
 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Embedding(nn.Module):
    def __init__(self, d_dim, embed_dim):
        super(Embedding, self).__init__()
        self.d_dim = d_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.d_dim, self.embed_dim)
        
    def forward(self, x):
        return self.embed(x)
    
    

def positional_embedding(sequence_length, embed_dim):
    '''
    Args:
        sequence_length: length of the input Sequence
        embed_dim: Embedding dimension
    '''
    pe = torch.zeros(sequence_length, embed_dim)
    for i in range(sequence_length):
        for j in range(embed_dim):
            pe[i][j] = np.sin(i / (10000 ** (j / embed_dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / embed_dim)))
            
    return pe

    


## Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        """ 
        Args:
            embed_dim: dimension of the embedding vector output
            n_heads: number of self attention heads
        """
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        assert self.embed_dim% self.n_heads == 0, 'n_heads is not a factor of Embed_dim '
        self.single_head_dim = int(self.embed_dim/self.n_heads)
        
        # Defining key, query, and value matrixes
        self.key_mat = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.query_mat = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.value_mat = nn.Linear(self.single_head_dim, self.single_head_dim, bias = False)
        self.W_o_mat = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim, bias = False)
        
        
    def forward (self, key, query, value, need_weights=False): # batch_size x sequence_length x embedding_dim
        ''' 
        Args: 
            key: key vector
            query: query vector
            value: value vector
            
        returns:
            output vectors from multihead attention
            
        '''
        batch_size = key.shape[0]
        sequence_length = key.shape[1]
        
        # Changing dimensions according to heads
        key = key.view(batch_size, sequence_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, sequence_length, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, sequence_length, self.n_heads, self.single_head_dim)
        
        # Computing key, query, and value vectors
        K = self.key_mat(key)
        Q = self.query_mat(query)
        V = self.value_mat(value)
        
        # Transposing the matrixes to find the multihead self attention
        
        K = K.transpose(1,2)
        Q = Q.transpose(1,2)
        V = V.transpose(1,2)
        
        # Attention computation
        # adjusting key for matrix multiplication
        K_adj = K.transpose(-1,-2) # (batch_size, n_heads, single_head_dim, sequence_length)
        
        product = Q@K_adj
        
        # scaling the product
        product = product/self.single_head_dim**0.5
        
        # normalizing the product
        scores = F.softmax(product, dim = -1)
        
        # Computing the self attention
        attn = scores@V
        
        # Concatenated output
        concat = attn.transpose(1,2).contiguous().view(batch_size, sequence_length, self.n_heads*self.single_head_dim)
   
        output = self.W_o_mat(concat)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(TransformerBlock, self).__init__()

        # First layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-head attention

        self.msa = MultiHeadAttention(embed_dim, n_heads)

        # Second layer Normalization
        self.norm2 = nn.LayerNorm((embed_dim))
        
        # Encoder Multilayer perceptron
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, embed_dim)
                                          ,nn.ReLU())
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x1):
        
        x = self.norm1(x1)
       
        attention = self.msa(x,x,x, need_weights=False) #  Multi-head attention
        attention = attention+x # Residual Connection
        out = self.dropout1(attention)
        
        out = self.norm2(out)
        out = self.feed_forward(out)+out # Feed-forward with residual connection
        
        return self.dropout2(out)



class TransformerEncoder(nn.Module):
    def __init__(self, sequence_length, embed_dim,d_dim, n_heads, num_layers, pos_enc):
        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_dim = d_dim
        self.sequence_length = sequence_length
        self.pos_enc = pos_enc
        
        #self.tokenizer = Tokenizer()
        self.embed = Embedding(self.d_dim, self.embed_dim)
        
        # positional Embedding
        
        
        self.layers = nn.ModuleList([TransformerBlock(self.embed_dim, self.n_heads)
                                    for i in range(num_layers)])
        
        self.fc1 = nn.Linear(self.sequence_length*self.embed_dim, self.embed_dim)
        self.dropout3 = nn.Dropout(0.1)
    def forward(self, x):
        #tokens = self.tokenizer(x)
        out = self.embed(x)
        if self.pos_enc:
            out += positional_embedding(self.sequence_length, self.embed_dim)
        
        for layer in self.layers:
            out = layer(out)
        # Attention based pooling
        #attn = self.attn(out)
        #attention_out = attn.transpose(2,1)@out
        
        # Avergae pooling
        representation = torch.flatten(out, start_dim=1)
        representation = self.fc1(representation)
        representation = self.dropout3(representation)
        return representation