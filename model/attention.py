import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        # Query, Key, Value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention Scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(k.size(-1))

        # ----------------------------
        # Causal Mask (IMPORTANT)
        # ----------------------------
        seq_len = x.size(1)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )

        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted Sum
        output = torch.matmul(attention_weights, v)

        return output