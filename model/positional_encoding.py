import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,embedding_dim,max_length=5000):
        super().__init__()

        position_encoding = torch.zeros(max_length,embedding_dim)
        position = torch.arange(0,max_length).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,embedding_dim,2)*(-math.log(1000.0)/embedding_dim))
        position_encoding[:,0::2] = torch.sin(position*div_term)
        position_encoding[:,1::2]=torch.cos(position*div_term)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer("position_encoding" , position_encoding)
    def forward(self,x):
        sq_length=x.size(1)
        return x + self.position_encoding[:,:sq_length]

    
