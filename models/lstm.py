#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:54:16 2023

@author: salim
"""

import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, dim, embed_dim):
        super(Embedding, self).__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.dim, self.embed_dim)

    def forward(self, x):
        return self.embed(x)
    

class LSTM(nn.Module):
    """
    A simple LSTM from Scratch
    """
    def __init__(self, sequence_length, dim, embed_dim, hidden_length):
        super(LSTM, self).__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.hidden_length = hidden_length
        self.embed = Embedding(dim, embed_dim)

        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.embed_dim, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.embed_dim, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.embed_dim, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()

        # out gate components
        self.linear_gate_w4 = nn.Linear(self.embed_dim, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()

    def forget(self, x, h):
        x = self.linear_forget_w1(x)
        h = self.linear_forget_r1(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x, h):
        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        return self.sigmoid_gate(x_temp + h_temp)

    def cell_memory_gate(self, i, f, x, h, c_prev):
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)

        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i

        # forget old context/cell info
        c = f * c_prev
        # learn new context/cell info
        c_next = g + c
        return c_next

    def out_gate(self, x, h):
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)

    def forward(self, x_t, tuple_in ):
        hidden_seq = []
        (h, c_prev) = tuple_in
        x_t = self.embed(x_t)
        for t in range(self.sequence_length):
            x = x_t[:,t,:]
            # Equation 1. input gate
            i = self.input_gate(x, h)

            # Equation 2. forget gate
            f = self.forget(x, h)

            # Equation 3. updating the cell memory
            c_next = self.cell_memory_gate(i, f, x, h,c_prev)

            # Equation 4. calculate the main output gate
            o = self.out_gate(x, h)

            # Equation 5. produce next hidden output
            h_next = o * self.activation_final(c_next)

            c_prev = c_next
            h = h_next

            hidden_seq.append(h.unsqueeze(0))

        #reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, h_next, c_next