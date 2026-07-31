import torch

from model.attention import SelfAttention
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward


# -----------------------------
# Create dummy input
# -----------------------------
batch_size = 1
sequence_length = 4
embedding_dim = 128
num_heads = 4
hidden_dim = 512

x = torch.randn(batch_size, sequence_length, embedding_dim)

print("Input Shape:")
print(x.shape)

# -----------------------------
# Test Self Attention
# -----------------------------
print("\n===== Self Attention =====")

self_attention = SelfAttention(embedding_dim)

self_output = self_attention(x)

print("Output Shape:")
print(self_output.shape)

print("First Token Output:")
print(self_output[0, 0])

# -----------------------------
# Test Multi Head Attention
# -----------------------------
print("\n===== Multi Head Attention =====")

multi_attention = MultiHeadAttention(
    embedding_dim,
    num_heads
)

multi_output = multi_attention(x)

print("Output Shape:")
print(multi_output.shape)

print("First Token Output:")
print(multi_output[0, 0])

# -----------------------------
# Test Feed Forward
# -----------------------------
print("\n===== Feed Forward =====")

feed_forward = FeedForward(
    embedding_dim,
    hidden_dim
)

ff_output = feed_forward(x)

print("Output Shape:")
print(ff_output.shape)

print("First Token Output:")
print(ff_output[0, 0])

# -----------------------------