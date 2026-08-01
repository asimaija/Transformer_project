import torch
import torch.nn as nn

from model.attention import SelfAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.2):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttention(self.head_dim)
            for _ in range(num_heads)
        ])

        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        outputs = []

        for i, head in enumerate(self.heads):

            start = i * self.head_dim
            end = (i + 1) * self.head_dim

            head_input = x[:, :, start:end]

            outputs.append(head(head_input))

        x = torch.cat(outputs, dim=-1)

        x = self.linear(x)

        x = self.dropout(x)

        return x