#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:38:51 2023

@author: salim
"""


import torch
import torch.nn.functional as F
from torch import nn, einsum
from functools import partial

from einops import rearrange, repeat




def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_empty(t: torch.Tensor):
    return t.numel() == 0



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

    
def l2norm(t):
    return F.normalize(t, dim = -1)


def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):  # dim=32
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, offset = 0):
        seq = torch.arange(max_seq_len) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')
    
# main class

class Attend(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):

        #b, n, device = q.shape[0], q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale


        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

# attention

class Attention(nn.Module):
    def __init__(self, dim_head, qk_rmsnorm = False, rotary_pos_emb=False):
        super().__init__()
        self.attend = Attend()
        self.qk_rmsnorm = qk_rmsnorm
        self.rotary_pos_emb = rotary_pos_emb
        self.rotary_pos_emb = RotaryEmbedding(dim_head) # rotary_emb_dim = 32


        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

    def forward(
        self,
        q, k, v
    ):

        seq_len = q.shape[-2]
        # rotary positional embedding with xpos for length extrapolation
        if self.qk_rmsnorm:
          #print('Normalization being done.....')

          q, k = map(l2norm, (q, k))
          q = q * self.q_scale
          k = k * self.k_scale

        if exists(self.rotary_pos_emb):
          #print('Embeeding being attached.....')
          attn_rotary_pos_emb = self.rotary_pos_emb(seq_len)
          q = apply_rotary_pos_emb(q, attn_rotary_pos_emb)
          k = apply_rotary_pos_emb(k, attn_rotary_pos_emb)

        # attention
        out = self.attend(q, k, v)

        return out
    
    
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_state_vectors,
        dim_head ,
        heads,
        qk_rmsnorm = False,
        rotary_pos_emb = False
    ):

        super().__init__()
        self.heads = heads                     # 8
        inner_dim = dim_head * heads           # 8*32=256
        self.state_norm = LayerNorm(dim)

        self.q_from_state = nn.Linear(dim, inner_dim, bias = False)      # (512, 256)
        self.k_from_state = nn.Linear(dim, inner_dim, bias = False)      # (512, 256)
        self.v_from_state = nn.Linear(dim, inner_dim, bias = False)      # (512, 256)

        self.to_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, rotary_pos_emb=rotary_pos_emb)

        self.from_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, rotary_pos_emb= rotary_pos_emb)
        self.state_out_to_gate = nn.Linear(dim, dim)


    def forward(self, q, k, v, state):

        # pre norm state for attention
        batch = q.shape[0]

        normed_states = self.state_norm(state)

        # get queries for cross attention, which they do not share, although they share key / values. another intriguing detail

        q_from_state = self.q_from_state(normed_states)
        k_from_state = self.k_from_state(normed_states)
        v_from_state = self.v_from_state(normed_states)

        # Rearranging the vectors
        q_from_state = rearrange(q_from_state, '... n (h d) -> ... h n d', h = self.heads)     # Q state shape:  torch.Size([1, 8, 199, 64])

        k_from_state = rearrange(k_from_state, '... n (h d) -> ... h n d', h = self.heads)

        v_from_state = rearrange(v_from_state, '... n (h d) -> ... h n d', h = self.heads)

        if q_from_state.ndim==3:
          q_from_state= repeat(q_from_state, 'h n d-> b h n d',  b = batch)
          k_from_state= repeat(k_from_state, 'h n d-> b h n d',  b = batch)
          v_from_state= repeat(v_from_state, 'h n d-> b h n d',  b = batch)

        # cross attend from past_query

        to_state_out = self.to_state_cross_attn(q_from_state, k, v)

        # cross attend to past states key values

        from_state_out = self.from_state_cross_attn(q, k_from_state, v_from_state)

        state_out = torch.cat((to_state_out, from_state_out), dim = -1)

        return state_out


class RecurrentStateGate(nn.Module):

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)

    def forward(self, x, state):
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)


class ARNN_Block(nn.Module):
    def __init__(
        self,
        embed_dim, dim_head, heads, num_state_vectors, qk_rmsnorm=False, rotary_pos_emb=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.to_q = nn.Linear(embed_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias = False)

        self.attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, rotary_pos_emb= rotary_pos_emb)

        self.attn_out = nn.Linear(inner_dim*3, embed_dim, bias = False)
        self.state_recurrence = RecurrentStateGate(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.cross_attention = CrossAttention(embed_dim, num_state_vectors, dim_head , heads, qk_rmsnorm = qk_rmsnorm, rotary_pos_emb=rotary_pos_emb)
        self.to_logits = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = SublayerConnection(embed_dim,0.1)
    def forward(
        self,
        x,
        state):

        q, k, v = (self.to_q(x),self.to_k(x), self.to_v(x))

        split_head = partial(rearrange, pattern = 'b n (h d) -> b h n d', h = self.heads)
        q = split_head(q)
        k = split_head(k)
        v = split_head(v)

        # attention, but of course

        self_out = self.attn(q, k, v) # self Attention

        cross_out = self.cross_attention(q,k,v, state) # cross Attention
        atten_out = torch.cat((cross_out, self_out), dim = -1)
        attention_out = self.attn_out(atten_out)
        next_state = self.state_recurrence(attention_out, state) # state_residual.shape torch.Size([1, 512, 400])
        output = self.norm(attention_out, self.to_logits) 

        return output, next_state

class ARNN(nn.Module):

    def __init__(self, embed_dim, seq_len, dim_head, heads, qk_rmsnorm, rotary_pos_emb, num_state_vectors, time_steps): #(self, 200, 400)
        super().__init__()
        #dim = int(d_dim/time_steps)
        self.norm_ex = LayerNorm(embed_dim)
        self.arnn = ARNN_Block(embed_dim, dim_head,  heads, num_state_vectors, qk_rmsnorm=qk_rmsnorm, rotary_pos_emb=rotary_pos_emb)
        self.init_state = nn.Parameter(torch.randn(num_state_vectors, embed_dim))
        self.time_steps = time_steps

    def forward(self, x):
        x = self.norm_ex(x)
        input_data = rearrange(x,'b (r n) d->  b r n d', r=self.time_steps)
        state = self.init_state
        for i in range(self.time_steps):
          x = input_data[:,i,:,:]
          x, state = self.arnn(x, state)
        return x, state