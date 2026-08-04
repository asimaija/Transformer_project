import requests
import torch

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# Unique characters
unique_characters = sorted(list(set(text)))


# Character mappings
stoi = {ch: i for i, ch in enumerate(unique_characters)}
itos = {i: ch for i, ch in enumerate(unique_characters)}

# Encode
def encode(text):
    return [stoi[ch] for ch in text]

# Decode
def decode(indices):
    return "".join([itos[i] for i in indices])

# Encode complete dataset
encoded_text = encode(text)



def split_data(encoded_text, train_ratio=0.9):
    split = int(len(encoded_text) * train_ratio)
    train_data = encoded_text[:split]
    val_data = encoded_text[split:]
    return train_data, val_data

train_data, val_data = split_data(encoded_text)

# Random batch
def get_batch(data, block_size, batch_size):

    max_index = len(data) - block_size - 1

    indices = torch.randint(
        0,
        max_index,
        (batch_size,)
    )

    x = torch.stack([
        torch.tensor(data[i:i + block_size], dtype=torch.long)
        for i in indices
    ])

    y = torch.stack([
        torch.tensor(data[i + 1:i + block_size + 1], dtype=torch.long)
        for i in indices
    ])

    return x, y