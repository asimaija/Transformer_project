import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.query = nn.Linear(embedding_dim,embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)
    def forward(self,x):
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        scores=torch.matmul(q,k.transpose(-2,-1))
        scores = scores/math.sqrt(q.size(-1))
        attention_weights = torch.softmax(scores , dim=-1)
        output = torch.matmul(attention_weights,v)
        return output