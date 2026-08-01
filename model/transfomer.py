import torch
import torch.nn as nn
from model.embedding import TokenEmbedding
from model.positional_encoding import PositionalEncoding
from model.transfomer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_heads,hidden_dim,num_layers,max_length=5000):
        super().__init__()
        self.embedding=TokenEmbedding(vocab_size,embedding_dim)
        self.position =PositionalEncoding(embedding_dim,max_length)
        self.blocks=nn.ModuleList([TransformerBlock(embedding_dim,num_heads,hidden_dim) for _ in range (num_layers)])
        self.fc = nn.Linear(embedding_dim,vocab_size)

    def forward(self,x):
        x=self.embedding(x)
        x=self.position(x)
        for block in self.blocks:
            x=block(x)
        x=self.fc(x)
        return x