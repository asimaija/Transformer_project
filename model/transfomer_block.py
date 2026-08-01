import torch
import torch.nn as nn
from model.feed_forward import FeedForward
from model.multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,hidden_dim):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim,num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim,hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    def forward(self,x):
        attention_output = self.attention(x)
        x=x+attention_output
        x=self.norm1(x)
        feed_forward_output = self.feed_forward(x)
        x=x+feed_forward_output
        x=self.norm2(x)
        return x