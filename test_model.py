import torch

from model.attention import SelfAttention
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.transfomer_block import TransformerBlock
from model.transfomer import Transformer

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 1
sequence_length = 4
embedding_dim = 128
num_heads = 4
hidden_dim = 512
vocab_size = 65
num_layers = 2

# -----------------------------
# Create Dummy Embedding Input
# -----------------------------
x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

print("=" * 50)
print("Input Shape:")
print(x.shape)

# =====================================================
# Self Attention
# =====================================================
print("\n===== Self Attention =====")

self_attention = SelfAttention(embedding_dim)

self_output = self_attention(x)

print("Output Shape:")
print(self_output.shape)

print("First Token Output:")
print(self_output[0, 0])

# =====================================================
# Multi Head Attention
# =====================================================
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

# =====================================================
# Feed Forward
# =====================================================
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

# =====================================================
# Transformer Block
# =====================================================
print("\n===== Transformer Block =====")

transformer_block = TransformerBlock(
    embedding_dim,
    num_heads,
    hidden_dim
)

block_output = transformer_block(x)

print("Output Shape:")
print(block_output.shape)

print("First Token Output:")
print(block_output[0, 0])

# =====================================================
# Complete Transformer
# =====================================================
print("\n===== Complete Transformer =====")

model = Transformer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)

# Transformer takes token IDs
input_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, sequence_length)
)

print("Input IDs Shape:")
print(input_ids.shape)

transformer_output = model(input_ids)

print("Output Shape:")
print(transformer_output.shape)

print("First Token Prediction:")
print(transformer_output[0, 0])

print("\n" + "=" * 50)
print("All Tests Passed Successfully!")
print("=" * 50)