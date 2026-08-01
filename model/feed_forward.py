import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x