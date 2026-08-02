import torch

from model.transfomer import Transformer
from data.dataset import encode, decode

vocab_size = 65
embedding_dim = 256
num_heads = 8
hidden_dim = 1024
num_layers = 4

block_size = 128
batch_size = 16

epochs = 10000
learning_rate = 3e-4

temperature = 0.8
max_new_tokens = 1

model = Transformer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)

model.load_state_dict(torch.load("transformer.pth"))
model.eval()

prompt = input("Enter starting text: ")

if len(prompt) == 0:
    prompt = "The "

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long
)

with torch.no_grad():

    for _ in range(max_new_tokens):

        # Keep only the last block_size tokens
        context_crop = context[:, -block_size:]

        # Forward pass
        logits = model(context_crop)

        # Last token prediction
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # Append token
        context = torch.cat(
            (context, next_token),
            dim=1
        )


generated_text = decode(context[0].tolist())

# Next predicted character
predicted_character = generated_text[len(prompt)]

# Word after adding the predicted character
generated_text = decode(context[0].tolist())

# First predicted character
predicted_character = generated_text[len(prompt)]

# Continue until the end of the current word
current_text = generated_text
while len(current_text) == len(prompt) or current_text[-1] not in [" ", "\n", ".", ",", ":", ";", "!", "?"]:

    context = torch.tensor([encode(current_text)], dtype=torch.long)

    with torch.no_grad():
        logits = model(context[:, -block_size:])
        logits = logits[:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    next_char = decode(next_token[0].tolist())
    current_text += next_char

predicted_word = current_text.split()[0]


print("Input               :", prompt)
print("Predicted Character :", predicted_character)
print("Predicted Word      :", predicted_word)
