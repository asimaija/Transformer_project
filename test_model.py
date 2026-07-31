import torch

from model.attention import SelfAttention
from model.multi_head_attention import MultiHeadAttention

# -----------------------------
# Create dummy input
# -----------------------------
batch_size = 1
sequence_length = 4
embedding_dim = 128
num_heads = 4

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