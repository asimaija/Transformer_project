import torch
import torch.nn as nn
from model.attention import SelfAttention
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(embedding_dim) for _ in range(num_heads)])
        self.linear = nn.Linear(embedding_dim*num_heads,embedding_dim)
    def forward(self,x):
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        output = torch.cat(outputs,dim=-1)
        output = self.linear(output)
        return output 