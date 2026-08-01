import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.2):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
            dropout
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Multi-Head Attention
        attention_output = self.attention(x)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Feed Forward
        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.norm2(x)

        return x