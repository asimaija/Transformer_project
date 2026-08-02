import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import train_data, get_batch
from model.transfomer import Transformer

vocab_size = 65
embedding_dim = 256
num_heads = 8
hidden_dim = 1024
num_layers = 4

block_size = 128
batch_size = 16

epochs = 10000
learning_rate = 3e-4

model = Transformer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):

    x, y = get_batch(
        train_data,
        block_size,
        batch_size
    )

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(
        output.view(-1, vocab_size),
        y.view(-1)
    )

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.4f}")


torch.save(model.state_dict(), "transformer.pth")

print("\nTraining Completed!")
print("Model Saved Successfully!")