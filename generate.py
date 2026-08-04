import torch

from model.transfomer import Transformer
from data.dataset import encode, decode

# Load trained model
model = Transformer(
    vocab_size=65,
    embedding_dim=256,
    num_heads=8,
    hidden_dim=1024,
    num_layers=4
)

model.load_state_dict(torch.load("transformer.pth"))
model.eval()

# User input
prompt = input("Enter text: ")

# Convert text to numbers
context = torch.tensor([encode(prompt)], dtype=torch.long)

# Predict 2 characters
with torch.no_grad():

    for i in range(1):         

        # Model prediction
        logits = model(context)

        # Take only last character prediction
        logits = logits[:, -1, :]

        # Convert scores into probabilities
        probs = torch.softmax(logits, dim=-1)

        # Pick one character
        next_token = torch.multinomial(probs, 1)

        # Add predicted character to input
        context = torch.cat((context, next_token), dim=1)

# Convert numbers back to text
generated_text = decode(context[0].tolist())

# Show only the new predicted characters
predicted = generated_text[len(prompt):]

print("Input      :", prompt)
print("Predicted  :", predicted)