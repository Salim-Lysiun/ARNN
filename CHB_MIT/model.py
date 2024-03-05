import os, sys
from torch import nn
from collections import OrderedDict
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'../models'))
from ARNN import ARNN

class Embedding(nn.Module):
    def __init__(self, d_dim, embed_dim):
        super(Embedding, self).__init__()
        self.d_dim = d_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(self.d_dim, self.embed_dim)
        
    def forward(self, x):
        return self.embed(x)
    
class MLP_layer(nn.Module):
  def __init__(self, embed_dim, rec_seq, class_len) -> None:
    super().__init__()
    self.sequential = nn.Sequential((OrderedDict([
          ('linear1', nn.Linear(embed_dim,embed_dim)),
          ('flat', nn.Flatten()),
          ('linear2', nn.Linear(embed_dim*rec_seq, class_len)),
          ('sigmoid', nn.Sigmoid())
        ]))) 
    
  def forward(self, x):
    return self.sequential(x)


class Attentive_RNN(nn.Module):
    def __init__(self, d_dim ,
    embed_dim ,
    seq_len ,
    dim_head,
    heads ,
    num_state_vectors,
    time_steps,
    num_class,
    qk_rmsnorm = True,
    rotary_pos_emb = True ,
    ):
        super(Attentive_RNN, self).__init__()
        rec_seq = seq_len//time_steps  # Dimension of vectors per recurrence layer
        self.embed = Embedding(d_dim, embed_dim)
        self.arnn = ARNN(embed_dim,
        seq_len ,
        dim_head,
        heads,
        qk_rmsnorm ,
        rotary_pos_emb,
        num_state_vectors,
        time_steps)
        
        self.mlp = MLP_layer(embed_dim, rec_seq, num_class)


    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.embed(x)
        out , state = self.arnn(out)
        return self.mlp(out)




