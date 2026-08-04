import torch
import torch.nn as nn

from model.attention import SelfAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        # Size of one head
        self.head_dim = embedding_dim // num_heads

        # Create multiple attention heads
        self.heads = nn.ModuleList([
            SelfAttention(self.head_dim)
            for _ in range(num_heads)
        ])

        # Final linear layer
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        outputs = []

        # Send part of embedding to each head
        for i, head in enumerate(self.heads):

            start = i * self.head_dim
            end = start + self.head_dim

            part = x[:, :, start:end]

            outputs.append(head(part))

        # Join all heads
        x = torch.cat(outputs, dim=-1)

        # Final linear layer
        x = self.linear(x)

        return x