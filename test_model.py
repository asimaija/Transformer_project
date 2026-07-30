import torch

from data.dataset import x, unique_characters
from model.embedding import TokenEmbedding
from model.positional_encoding import PositionalEncoding
x = torch.tensor([x])
vocab_size = len(unique_characters)
embedding_dim = 128

embedding = TokenEmbedding(vocab_size,embedding_dim)
embedded=embedding(x)

Position = PositionalEncoding(embedding_dim)
output = Position(embedded)
 


print("Input IDs" ,x)
print("Embedding shape",embedded.shape)

print("Output shape",output.shape)
print("First character embedding",embedded[0,0])
print("First positional emdding",output[0,0])
